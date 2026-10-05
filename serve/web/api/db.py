"""数据库管理 API（读写）。"""
import os
import hashlib
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List, Any
from serve.memory import db_manager as dm

router = APIRouter()


@router.get("/list")
def list_dbs():
    return {"dbs": dm.list_dbs()}


@router.get("/fingerprint")
def fingerprint():
    """数据指纹（mtime + size）。前端轮询判断是否变化。"""
    out = {}
    for name, path in dm.DBS.items():
        if os.path.exists(path):
            st = os.stat(path)
            # WAL 模式下主 db 文件 mtime 变化延迟，用 mtime + size
            out[name] = f"{int(st.st_mtime)}:{st.st_size}"
        else:
            out[name] = "0:0"
    return {"fp": out}


@router.get("/{db_name}/tables")
def list_tables(db_name: str):
    try:
        tables = dm.list_tables(db_name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"db": db_name, "tables": tables}


@router.get("/{db_name}/table/{table}")
def query_table(db_name: str, table: str,
                offset: int = 0, limit: int = 50,
                order_by: Optional[str] = None, desc: bool = True):
    try:
        return dm.fetch_rows(db_name, table, offset, limit,
                             order_by, desc)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/{db_name}/table/{table}/insert")
def insert_row(db_name: str, table: str, row: dict = Body(...)):
    cols = dm.table_columns(db_name, table)
    use = [c for c in cols if c in row]
    if not use:
        raise HTTPException(400, "无有效字段")
    sql = (f"INSERT INTO `{table}` "
           f"({', '.join(f'`{c}`' for c in use)}) "
           f"VALUES ({', '.join('?' for _ in use)})")
    try:
        rid = dm.execute(db_name, sql, tuple(row[c] for c in use))
        return {"ok": True, "lastrowid": rid}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.put("/{db_name}/table/{table}/update")
def update_row(db_name: str, table: str,
               pk: str = Query(...), pk_val: Any = Query(...),
               row: dict = Body(...)):
    cols = dm.table_columns(db_name, table)
    if pk not in cols:
        raise HTTPException(400, f"主键列不存在: {pk}")
    use = [c for c in cols if c in row and c != pk]
    if not use:
        raise HTTPException(400, "无有效字段")
    sql = (f"UPDATE `{table}` SET "
           + ", ".join(f"`{c}`=?" for c in use)
           + f" WHERE `{pk}`=?")
    try:
        n = dm.execute(db_name, sql, tuple(row[c] for c in use) + (pk_val,))
        return {"ok": True, "affected": n}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.delete("/{db_name}/table/{table}/delete")
def delete_row(db_name: str, table: str,
               pk: str = Query(...), pk_val: Any = Query(...)):
    sql = f"DELETE FROM `{table}` WHERE `{pk}`=?"
    try:
        n = dm.execute(db_name, sql, (pk_val,))
        return {"ok": True, "affected": n}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/{db_name}/sql")
def raw_sql(db_name: str, body: dict = Body(...)):
    sql = (body.get("sql") or "").strip()
    if not sql:
        raise HTTPException(400, "sql 为空")
    mode = body.get("mode", "query")
    params = tuple(body.get("params") or ())

    lowered = sql.lower()
    for kw in ("drop table", "drop database", "attach ", "detach "):
        if kw in lowered:
            raise HTTPException(403, f"禁止操作: {kw}")

    try:
        if mode == "query":
            rows = dm.query(db_name, sql, params)
            return {"ok": True, "rows": rows[:500]}
        else:
            n = dm.execute(db_name, sql, params)
            return {"ok": True, "affected": n}
    except Exception as e:
        raise HTTPException(400, str(e))

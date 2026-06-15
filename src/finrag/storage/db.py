"""存储适配器共享的轻量数据库辅助层"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class Database:
    """封装数据库连接创建和事务提交回滚"""

    def __init__(self, database_url: str):
        """
        初始化数据库辅助对象
        Args:
            database_url: PostgreSQL 数据库连接串
        """
        self.database_url = database_url

    @contextmanager
    def connect(self) -> Iterator[Any]:
        """
        创建数据库连接并在上下文退出时提交或回滚
        Yields:
            数据库连接对象
        """
        import psycopg

        connection = psycopg.connect(self.database_url)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def execute(connection: Any, sql: str, params: tuple[Any, ...] = ()):
    """
    执行单条 PostgreSQL SQL
    Args:
        connection: 数据库连接对象
        sql: 使用 psycopg %s 占位符的 SQL 语句
        params: SQL 参数元组
    Returns:
        数据库驱动返回的执行结果
    """
    return connection.execute(sql, params)


def executemany(connection: Any, sql: str, params):
    """
    批量执行 PostgreSQL SQL
    Args:
        connection: 数据库连接对象
        sql: 使用 psycopg %s 占位符的 SQL 语句
        params: 多组 SQL 参数
    Returns:
        数据库驱动返回的执行结果
    """
    if hasattr(connection, "executemany"):
        return connection.executemany(sql, params)
    with connection.cursor() as cursor:
        return cursor.executemany(sql, params)

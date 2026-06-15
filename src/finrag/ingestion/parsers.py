"""FinRAG 通用金融文档解析入口"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from llama_index.core import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"} # 支持解析和上传的文件后缀
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "utf-16-le", "utf-16-be") # 文本文件候选编码顺序


def utc_now_iso() -> str:
    """
    获取当前 UTC 时间的 ISO 字符串
    Returns:
        带时区信息的 ISO 时间字符串
    """
    return datetime.now(timezone.utc).isoformat()


def compute_content_hash(path: Path) -> str:
    """
    计算文件内容 SHA256 哈希，用于文档去重
    Args:
        path: 待计算的文件路径
    Returns:
        sha256: 前缀的内容哈希
    """
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    """
    规范化文档文本，统一换行、空白和空行
    Args:
        text: 原始文本
    Returns:
        清洗后的文本
    """
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines: List[str] = []
    blank = False # 上一行是否为空行
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip() # 统一空白为单个空格并去除行首尾空白
        if line:
            lines.append(line)
            blank = False
        elif not blank: # 当前行为空行且上一行不是空行
            lines.append("") # 保留一个空行
            blank = True
    return "\n".join(lines).strip()


def read_text_file(path: Path) -> str:
    """
    按常见编码顺序读取文本文件
    Args:
        path: 文本文件路径
    Returns:
        读取到的文本；编码无法识别时返回空字符串
    """
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    logger.warning("文本文件编码无法识别: %s", path)
    return ""


def _relative_path(path: Path, data_root: Optional[Path]) -> str:
    """
    计算文件相对数据根目录的路径，失败时返回绝对路径
    Args:
        path: 文件路径
        data_root: 数据根目录
    Returns:
        POSIX 风格路径字符串
    """
    if data_root is not None:
        try:
            return path.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_posix()


def is_path_within(path: Path, root: Path) -> bool:
    """
    判断路径是否位于指定根目录内
    Args:
        path: 待判断路径
        root: 根目录
    Returns:
        路径在根目录内时返回 True
    """
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_base_metadata(
    path: Path,
    text: str,
    file_type: str,
    *,
    knowledge_base_id: str = "default",
    page_number: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> dict:
    """
    为解析出的 Document 构造统一基础元数据
    Args:
        path: 源文件路径
        text: 文档文本
        file_type: 文件类型
        knowledge_base_id: 资料库 ID
        page_number: PDF 页码
        data_root: 数据根目录
    Returns:
        包含 document_id、文件名和页码的元数据字典
    """
    content_hash = compute_content_hash(path)
    relative_path = _relative_path(path, data_root)
    document_seed = f"{knowledge_base_id}:{relative_path}:{content_hash}"
    if page_number is not None:
        document_seed += f":page:{page_number}"
    # 用 MD5 哈希种子计算文档 ID，确保唯一性
    document_id = hashlib.md5(document_seed.encode("utf-8")).hexdigest()
    return {
        "knowledge_base_id": knowledge_base_id,
        "document_id": document_id,
        "source_path": path.resolve().as_posix(),
        "filename": path.name,
        "file_type": file_type,
        "page_number": page_number,
    }


class ParserRegistry:
    """文档解析器注册表，将文件后缀映射到解析函数"""

    def __init__(self):
        """
        初始化空解析器注册表
        """
        self._parsers: Dict[str, Callable[..., List[Document]]] = {}

    @classmethod
    def default(cls) -> "ParserRegistry":
        """
        创建注册了默认文档解析器的 ParserRegistry
        Returns:
            支持 md、txt、pdf、docx 的解析器注册表
        """
        registry = cls()
        registry.register(".md", parse_text_like)
        registry.register(".txt", parse_text_like)
        registry.register(".pdf", parse_pdf)
        registry.register(".docx", parse_docx)
        return registry

    def register(self, suffix: str, parser: Callable[..., List[Document]]) -> None:
        """
        注册指定文件后缀的解析函数
        Args:
            suffix: 文件后缀，如 .pdf
            parser: 返回 Document 列表的解析函数
        Returns:
            无返回值
        """
        self._parsers[suffix.lower()] = parser

    def load(
        self,
        path: Path,
        *,
        knowledge_base_id: str = "default",
        data_root: Optional[Path] = None,
    ) -> List[Document]:
        """
        根据文件后缀选择解析器并加载文档
        Args:
            path: 待解析文件路径
            knowledge_base_id: 资料库 ID
            data_root: 数据根目录
        Returns:
            解析得到的 Document 列表；不支持的文件返回空列表
        """
        parser = self._parsers.get(path.suffix.lower()) # 获取文件后缀对应的解析函数
        if parser is None:
            return []
        return parser(path, knowledge_base_id=knowledge_base_id, data_root=data_root)


def parse_text_like(
    path: Path,
    *,
    knowledge_base_id: str,
    data_root: Optional[Path] = None,
) -> List[Document]:
    """
    解析 Markdown/TXT 等纯文本类文档
    Args:
        path: 文件路径
        knowledge_base_id: 资料库 ID
        data_root: 数据根目录
    Returns:
        包含全文文本和元数据的 Document 列表
    """
    # 读取和规范化文本内容
    text = normalize_text(read_text_file(path))
    if not text:
        return []
    file_type = path.suffix.lower().lstrip(".")
    return [
        Document(
            text=text,
            metadata=build_base_metadata(
                path,
                text,
                file_type,
                knowledge_base_id=knowledge_base_id,
                data_root=data_root,
            ),
        )
    ]


def parse_pdf(
    path: Path,
    *,
    knowledge_base_id: str,
    data_root: Optional[Path] = None,
) -> List[Document]:
    """
    按页解析 PDF 文档，并为每页生成独立 Document
    Args:
        path: PDF 文件路径
        knowledge_base_id: 资料库 ID
        data_root: 数据根目录
    Returns:
        每个非空页面对应的 Document 列表
    """
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # 捕获所有异常，包括 malformed PDFs
        logger.warning("PDF 读取失败: %s, error=%s", path, exc)
        return []
    pages = []
    for page_number, page in enumerate(reader.pages, 1):
        # 提取页面文本并规范化
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        pages.append((page_number, text))
    return [
        Document(
            text=text,
            metadata=build_base_metadata(
                path,
                text,
                "pdf",
                knowledge_base_id=knowledge_base_id,
                page_number=page_number,
                data_root=data_root,
            ),
        )
        for page_number, text in pages
    ]


def parse_docx(
    path: Path,
    *,
    knowledge_base_id: str,
    data_root: Optional[Path] = None,
) -> List[Document]:
    """
    解析 DOCX 文档段落文本
    Args:
        path: DOCX 文件路径
        knowledge_base_id: 资料库 ID
        data_root: 数据根目录
    Returns:
        包含全文文本和元数据的 Document 列表
    """
    try:
        from docx import Document as DocxDocument
    except Exception as exc:  # 捕获所有异常，包括依赖缺失
        logger.warning("DOCX 解析依赖不可用: %s", exc)
        return []
    try:
        docx = DocxDocument(str(path))
    except Exception as exc:  # 捕获所有异常，包括 malformed DOCXs
        logger.warning("DOCX 读取失败: %s, error=%s", path, exc)
        return []
    # 提取段落文本并规范化
    text = normalize_text("\n".join(paragraph.text for paragraph in docx.paragraphs))
    if not text:
        return []
    return [
        Document(
            text=text,
            metadata=build_base_metadata(
                path,
                text,
                "docx",
                knowledge_base_id=knowledge_base_id,
                data_root=data_root,
            ),
        )
    ]


def load_documents(
    data_path: str | Path,
    *,
    knowledge_base_id: str = "default",
    document_registry: Optional[object] = None,
    parser_registry: Optional[ParserRegistry] = None,
) -> List[Document]:
    """
    从数据目录加载支持格式文档，或按文档注册表加载未删除文档
    Args:
        data_path: 数据根目录
        knowledge_base_id: 默认资料库 ID
        document_registry: 可选文档生命周期注册表
        parser_registry: 可选解析器注册表
    Returns:
        解析得到的 LlamaIndex Document 列表
    """
    root = Path(data_path)
    # 初始化解析器注册表
    registry = parser_registry or ParserRegistry.default()
    if not root.exists():
        logger.warning("文档目录不存在: %s", root)
        return []
    docs: List[Document] = []
    if document_registry is not None:
        records = []
        for record in document_registry.records.values():
            path = Path(record.source_path)
            # 跳过已删除或文件不存在的注册文档
            if record.status == "deleted" or not path.exists():
                continue
            # 跳过可信目录外的注册文档
            if not is_path_within(path, root):
                logger.warning("跳过可信目录外的注册文档: %s", path)
                continue
            records.append(record)
        # 按上传时间排序后加载注册文档，确保先上传的文档先被加载和解析
        for record in sorted(records, key=lambda item: item.upload_time):
            path = Path(record.source_path)
            # 仅加载支持格式的注册文档
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                # 加载注册文档
                parsed_docs = registry.load(
                    path,
                    knowledge_base_id=record.knowledge_base_id,
                    data_root=root,
                )
                for doc in parsed_docs:
                    # 将注册表中的文档元数据更新到解析得到的 Document 中，确保与注册表一致
                    doc.metadata.update(
                        {
                            "knowledge_base_id": record.knowledge_base_id,
                            "document_id": record.document_id,
                            "filename": record.filename,
                            "file_type": record.file_type,
                        }
                    )
                docs.extend(parsed_docs)
        return docs
    # 无注册表时按文件系统加载，可能包含已删除或未注册的文档
    for path in sorted(root.rglob("*")): # 递归遍历数据根目录
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(
                registry.load(
                    path,
                    knowledge_base_id=knowledge_base_id,
                    data_root=root,
                )
            )
    return docs


@dataclass
class DocumentRecord:
    """文档生命周期注册记录"""

    document_id: str # 文档 ID
    source_path: str # 源文件路径
    filename: str # 文件名
    file_type: str # 文件类型，如 pdf、md
    content_hash: str # 文件内容哈希值
    knowledge_base_id: str # 资料库 ID
    status: str = "uploaded" # 文档状态，如 uploaded、parsed、deleted
    chunk_count: int = 0 # 该文档生成的叶子分块数量
    upload_time: str = "" # 文档上传时间 ISO 字符串
    last_error: Optional[str] = None # 最新解析错误信息

    def to_dict(self) -> dict:
        """
        将文档记录转换为可 JSON 序列化字典
        Returns:
            文档生命周期记录字典
        """
        return asdict(self)

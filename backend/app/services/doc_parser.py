"""文档解析与切块。

职责：把上传的文件变成「文本块 + 章节归属」，供向量化入库。
支持的格式：.docx（python-docx）、.pdf（pypdf）、.txt/.md（内置读取）。
"""
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# 章节标题识别：第X章 / 第X条 / 一、 等常见中文标题格式
CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十零百]+章")
ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十零百]+条")


def _is_heading(text: str) -> bool:
    return bool(CHAPTER_PATTERN.match(text) or ARTICLE_PATTERN.match(text))


def parse_docx(path: Path) -> list[dict]:
    import docx

    document = docx.Document(str(path))
    blocks: list[dict] = []
    chapter = ""
    article = ""

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 更新章节上下文
        if CHAPTER_PATTERN.match(text):
            chapter = text
            article = ""
        elif ARTICLE_PATTERN.match(text):
            article = text

        blocks.append(
            {
                "text": text,
                "section": " · ".join(x for x in (chapter, article) if x),
                "page": None,
            }
        )
    return blocks


def parse_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[dict] = []
    chapter = ""

    for page_no, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            if CHAPTER_PATTERN.match(text):
                chapter = text
            blocks.append({"text": text, "section": chapter, "page": page_no})
    return blocks


def parse_text(path: Path) -> list[dict]:
    """纯文本/Markdown：按行读，尽量识别章节标题。"""
    content = ""
    for encoding in ("utf-8", "gbk"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if not content:
        raise ValueError(f"无法解码文件（试过 utf-8 / gbk）：{path.name}")

    blocks: list[dict] = []
    chapter = ""
    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        if CHAPTER_PATTERN.match(text.lstrip("# ")):
            chapter = text.lstrip("# ").strip()
        blocks.append({"text": text, "section": chapter, "page": None})
    return blocks


PARSERS = {".docx": parse_docx, ".pdf": parse_pdf, ".txt": parse_text, ".md": parse_text}


def parse_file(path: Path) -> list[dict]:
    """按扩展名选择解析器。"""
    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"暂不支持的文件类型：{suffix}（支持 docx / pdf / txt / md）")
    return parser(path)


def split_blocks(blocks: list[dict]) -> list[dict]:
    """把段落合并成 500 字左右的知识块，并保留章节、页码等元数据。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=True,
    )

    chunks: list[dict] = []
    index = 0
    for block in blocks:
        pieces = splitter.split_text(block["text"]) or [block["text"]]
        for piece in pieces:
            text = piece.strip()
            if len(text) < 5:  # 过短的碎片直接丢弃，避免污染检索
                continue
            chunks.append(
                {
                    "text": text,
                    "section": block.get("section", ""),
                    "page": block.get("page"),
                    "index": index,
                }
            )
            index += 1
    return chunks


def build_embed_texts(chunks: list[dict]) -> list[str]:
    """拼出用于向量化的文本：把章节标题带上，提升「第X章讲什么」这类问题的命中率。"""
    texts = []
    for chunk in chunks:
        section = chunk.get("section") or ""
        texts.append(f"【{section}】\n{chunk['text']}" if section else chunk["text"])
    return texts

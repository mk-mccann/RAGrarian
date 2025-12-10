from typing import Dict
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared citation formatting helpers (reusable across chat/query/UI)
# ---------------------------------------------------------------------------

def format_header_chain(metadata: dict) -> str | None:
    """
    Build a hierarchical header chain H1 > H2 > H3 > H4 from metadata.
    Falls back to common variants (head_*) and a generic 'header' field.
    Returns None if nothing present.

    Args:
        metadata (dict): Document metadata
    Returns:
        str | None: Formatted header chain or None
    """

    levels_primary = ['header_1', 'header_2', 'header_3', 'header_4']
    levels_alt = ['head_1', 'head_2', 'head_3', 'head_4']
    vals: list[str] = []

    for k in levels_primary:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
    if not vals:
        for k in levels_alt:
            v = metadata.get(k)
            if isinstance(v, str) and v.strip():
                vals.append(v.strip())
    if not vals:
        v = metadata.get('header')
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    vals = [h for h in vals if not (h in seen or seen.add(h))]
    if not vals:
        return None
    return " > ".join(vals) if len(vals) > 1 else vals[0]


def build_citation(doc: Document, 
                   source_number: int, 
                   doc_score: float | None = None
                   ) -> dict:
    """
    Create a structured citation dict from a Document for consistent use.
    Returns keys commonly used by the UI and logs:
    - source_number, title, header, url, page, file, file_path, metadata, citation_text

    Args:
        doc (Document): Document to build citation from
        source_number (int): Number to assign to this source
        doc_score (float | None): Optional document similarity score

    Returns:
        dict: Citation information
    """

    md = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
    
    # Normalize string "None" to actual None and decode bytes to str
    for key in list(md.keys()):
        if isinstance(md[key], bytes):
            md[key] = md[key].decode('utf-8', errors='ignore')
        if md[key] == "None":
            md[key] = None


    source = md.get('source') or 'Unknown'
    title = md.get('title') or 'Unknown'
    header = format_header_chain(md)
    url = md.get('url', None)
    page = md.get('page', None)
    file_label = md.get('file_path') or md.get('path') or 'Unknown'
    file_path = md.get('file_path')    

    citation_text = f"[{source_number}] {source}: {title}, "
    if header:
        citation_text += f"Section: {header}, "
    if url is not None:
        citation_text += f"URL: {url}, "
    if page is not None:
        citation_text += f"Page: {page}, "

    if url is not None:
        hyperlink_citation = f"[{source_number}] [{source}: {title}]({url})"
    else:
        hyperlink_citation = None

    if doc_score is not None:
        score = f'{doc_score:.2f}'
    else:
        score = "N/A"

    # Trim any trailing comma + space and close paren
    citation_text = citation_text.rstrip(', ')

    return {
        "source_number": source_number,
        "source": source,
        "title": title,
        "header": header,
        "url": url,
        "page": page,
        "file": file_label,
        "file_path": file_path,
        "metadata": md,
        "score": score,
        "citation_text": citation_text,
        "hyperlink_citation": hyperlink_citation
    }


def format_citation_line(citation: Dict[str, str], 
                         include_content: str | None = None,
                         debug_score: bool = False
                         ) -> str:
    """
    Render a single citation line (optionally followed by content).

    Args:
        citation (Dict[str, str]): Citation dictionary from build_citation().
        include_content (str | None): if provided, appended after the citation line separated by newline.
        debug_score (bool): If True, includes the document score in the citation. Not used if score is absent.

    Returns:
        str: Formatted citation line (and content if provided).
    """

    base = citation.get("hyperlink_citation", citation.get("citation_text", ""))

    if debug_score:
        base += f" [Score: {citation['score']}]"
   
    if include_content:
        return f"{base}\n{include_content}"
    else:
        return base


def format_context_for_display(retrieved_docs: list[tuple[Document, float | None]], 
                               debug_score: bool = False) -> list[str]:
    """
    Format retrieved documents as citation strings for display.
    
    Args:
        retrieved_docs: List of (Document, score) tuples.
        
    Returns:
        List of formatted citation strings.
    """
    context = []
    for idx, (doc, score) in enumerate(retrieved_docs, 1):
        citation_dict = build_citation(doc, idx, score)
        citation = format_citation_line(citation_dict, debug_score=debug_score)
        context.append(citation)
    return context

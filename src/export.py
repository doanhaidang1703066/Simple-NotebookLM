from typing import Literal

ExportFormat = Literal["text", "md", "json"]

from schemas import Summary, QuizSet, FlashcardSet

def _to_markdown(result_obj) -> str:
    """
    Translates learning objects (Summary, QuizSet, FlashcardSet)
    into a well-formatted Markdown string for UI display or file export.
    """
    md_lines = []

    # -----------------------------------------
    # FORMATTING FOR SUMMARY
    # -----------------------------------------
    if isinstance(result_obj, Summary):
        md_lines.append("## Summary")
        md_lines.append(f"**Scope:** {result_obj.target} *(Type: {result_obj.scope})*\n")
        md_lines.append(f"{result_obj.summary}\n")
        
        if result_obj.key_points:
            md_lines.append("### 💡 Key Points:")
            for kp in result_obj.key_points:
                md_lines.append(f"- {kp}")
                
    # -----------------------------------------
    # FORMATTING FOR QUIZ
    # -----------------------------------------
    elif isinstance(result_obj, QuizSet):
        md_lines.append("## Quiz Set")
        md_lines.append(f"**Scope:** {result_obj.target} *(Type: {result_obj.scope})*\n")
        
        for i, item in enumerate(result_obj.items, start=1):
            md_lines.append(f"**Question {i}: {item.question}**")
            for opt in item.options:
                md_lines.append(f"- {opt}")
            md_lines.append(f"\n* Answer:* **{item.answer}**")
            md_lines.append(f"*📖 Explanation:* {item.explanation}")
            
            # Display source markers (e.g., [S1, S2])
            if item.source_markers:
                md_lines.append(f"*🔗 Citations:* [{', '.join(item.source_markers)}]\n")
            md_lines.append("---")
            
    # -----------------------------------------
    # FORMATTING FOR FLASHCARDS
    # -----------------------------------------
    elif isinstance(result_obj, FlashcardSet):
        md_lines.append("## Flashcards")
        md_lines.append(f"**Scope:** {result_obj.target} *(Type: {result_obj.scope})*\n")
        
        for i, card in enumerate(result_obj.cards, start=1):
            md_lines.append(f"### Card {i}")
            md_lines.append(f"**Front (Q):** {card.front}")
            md_lines.append(f"**Back (A):** {card.back}")
            if card.source_markers:
                md_lines.append(f"*🔗 Citations:* [{', '.join(card.source_markers)}]\n")
            md_lines.append("---")

    else:
        raise ValueError("Unsupported object type for Markdown conversion.")

    # -----------------------------------------
    # APPEND REFERENCE LIST AT THE BOTTOM
    # -----------------------------------------
    if hasattr(result_obj, 'citations') and result_obj.citations:
        md_lines.append("\n### Detailed References:")
        for cite in result_obj.citations:
            md_lines.append(f"- **[{cite.source_marker}]**: File `{cite.filename}`, Page {cite.page}")

    # Join all lines with a newline character for the final string
    return "\n".join(md_lines)


def export(model, *, fmt="text", output=None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")
    
    if output is None:
        return text
    
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
from pathlib import Path

from src.preprocess import clean_review, load_dataset, preprocess_dataframe


def test_clean_review_removes_html_and_urls() -> None:
    text = "<p>Visit https://example.com for great films!</p>"
    cleaned = clean_review(text)
    assert "<p>" not in cleaned
    assert "https" not in cleaned
    assert "great" in cleaned or "film" in cleaned


def test_preprocess_dataframe(tmp_path: Path) -> None:
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "review,sentiment\n<p>Awesome film!</p>,1\n<p>Awesome film!</p>,1\n"
    )
    df = load_dataset(str(csv_file))
    assert len(df) == 1
    processed = preprocess_dataframe(df)
    assert "<p>" not in processed["review"].iloc[0]

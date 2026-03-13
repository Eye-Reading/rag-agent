"""
PDF 렌더러

마크다운 문자열을 전문 투자 보고서 스타일의 PDF로 변환합니다.

파이프라인:
  markdown string → HTML (markdown 라이브러리) → styled HTML → PDF (weasyprint)

폰트:
  Apple SD Gothic Neo (macOS 시스템 폰트, 한/영 혼용 최적)
  폴백: Noto Sans KR → sans-serif
"""
import ctypes
import os
import re
from datetime import date

# macOS Homebrew 환경에서 weasyprint가 libgobject를 찾지 못하는 경우를 대비해
# /opt/homebrew/lib 경로를 DYLD_LIBRARY_PATH에 선제 등록합니다.
_HOMEBREW_LIB = "/opt/homebrew/lib"
if os.path.isdir(_HOMEBREW_LIB):
    os.environ["DYLD_LIBRARY_PATH"] = (
        _HOMEBREW_LIB + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
    ).rstrip(":")
    # cffi가 dlopen을 캐싱하기 전에 직접 로드해 경로를 확정합니다.
    try:
        ctypes.CDLL(os.path.join(_HOMEBREW_LIB, "libgobject-2.0.0.dylib"))
    except OSError:
        pass

import markdown
from weasyprint import HTML, CSS


# ──────────────────────────────────────────
# CSS — 전문 투자 보고서 스타일
# ──────────────────────────────────────────

_REPORT_CSS = CSS(string="""
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

@font-face {
    font-family: 'AppleSDGothic';
    src: local('Apple SD Gothic Neo'), local('AppleSDGothicNeo');
}

@page {
    size: A4;
    margin: 22mm 20mm 25mm 22mm;

    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: 'AppleSDGothic', 'Noto Sans KR', sans-serif;
        font-size: 9pt;
        color: #94a3b8;
    }
    @bottom-left {
        content: "CONFIDENTIAL — Investment Evaluation Report";
        font-family: 'AppleSDGothic', 'Noto Sans KR', sans-serif;
        font-size: 8pt;
        color: #cbd5e1;
        font-style: italic;
    }
}

/* ── 기본 타이포그래피 ── */
body {
    font-family: 'AppleSDGothic', 'Noto Sans KR', sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1e293b;
    background: #ffffff;
    word-break: keep-all;
    overflow-wrap: break-word;
}

/* ── 커버 헤더 (h1) ── */
h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #ffffff;
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    margin: -22mm -20mm 12mm -22mm;
    padding: 18mm 22mm 12mm 22mm;
    letter-spacing: -0.5pt;
    page-break-after: avoid;
    border-bottom: 4px solid #f59e0b;
}

/* ── 섹션 제목 (h2) ── */
h2 {
    font-size: 13.5pt;
    font-weight: 700;
    color: #1e3a5f;
    border-left: 5px solid #2563eb;
    padding: 4pt 0 4pt 12pt;
    margin: 18pt 0 8pt 0;
    background: #f0f6ff;
    page-break-after: avoid;
}

/* ── 서브섹션 (h3) ── */
h3 {
    font-size: 11.5pt;
    font-weight: 700;
    color: #1d4ed8;
    border-bottom: 1.5px solid #bfdbfe;
    padding-bottom: 3pt;
    margin: 14pt 0 6pt 0;
    page-break-after: avoid;
}

/* ── 소제목 (h4) ── */
h4 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #334155;
    margin: 10pt 0 4pt 0;
    page-break-after: avoid;
}

/* ── 단락 ── */
p {
    margin: 0 0 7pt 0;
    text-align: justify;
}

/* ── 표 ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 14pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead tr {
    background: #1e3a5f;
    color: #ffffff;
}

thead th {
    padding: 7pt 10pt;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.2pt;
    border: 1px solid #1e3a5f;
}

tbody tr:nth-child(odd) {
    background: #f8fafc;
}

tbody tr:nth-child(even) {
    background: #ffffff;
}

tbody tr:hover {
    background: #eff6ff;
}

tbody td {
    padding: 6pt 10pt;
    border: 1px solid #e2e8f0;
    vertical-align: top;
}

/* 첫 번째 열 강조 */
tbody td:first-child {
    font-weight: 500;
    color: #1e3a5f;
}

/* ── 인용 블록 (리스크 섹션 등) ── */
blockquote {
    background: #fff7ed;
    border-left: 4px solid #f59e0b;
    margin: 8pt 0;
    padding: 8pt 14pt;
    border-radius: 0 4pt 4pt 0;
    page-break-inside: avoid;
}

blockquote p {
    margin: 3pt 0;
    color: #78350f;
    font-size: 9.5pt;
}

/* ── 목록 ── */
ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 18pt;
}

li {
    margin-bottom: 3pt;
    line-height: 1.6;
}

/* ── 굵은 글씨 ── */
strong {
    font-weight: 700;
    color: #0f172a;
}

/* ── 강조 ── */
em {
    color: #2563eb;
    font-style: normal;
    font-weight: 500;
}

/* ── 인라인 코드 ── */
code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 3pt;
    padding: 1pt 4pt;
    font-size: 9pt;
    color: #0f172a;
    font-family: 'AppleSDGothic', 'Noto Sans KR', monospace;
}

/* ── 수평선 ── */
hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 14pt 0;
}

/* ── SUMMARY 박스 강조 ── */
.summary-box {
    background: #eff6ff;
    border: 1.5px solid #93c5fd;
    border-radius: 6pt;
    padding: 12pt 16pt;
    margin: 10pt 0 16pt 0;
}

/* ── 페이지 나눔 방지 ── */
h2 + table, h3 + table, h2 + ul, h3 + ul {
    page-break-before: avoid;
}
""")


# ──────────────────────────────────────────
# SUMMARY 섹션 박스 처리
# ──────────────────────────────────────────

def _wrap_summary_section(html: str) -> str:
    """
    SUMMARY 섹션 바로 다음에 오는 <p> 블록들을 강조 박스로 감쌉니다.
    h2#summary 다음 ~ 다음 h2 전까지의 p 태그 그룹을 대상으로 합니다.
    """
    # SUMMARY 헤딩 바로 뒤 단락들에 summary-box div를 적용합니다.
    pattern = re.compile(
        r'(<h2[^>]*>SUMMARY</h2>)\s*(<p>.*?</p>(?:\s*<p>.*?</p>)*)',
        re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub(
        lambda m: m.group(1) + '\n<div class="summary-box">' + m.group(2) + '</div>',
        html,
    )


# ──────────────────────────────────────────
# 마크다운 → HTML 변환
# ──────────────────────────────────────────

_MD_EXTENSIONS = [
    "tables",       # | 파이프 표
    "fenced_code",  # ``` 코드 블록
    "nl2br",        # 줄바꿈 → <br>
    "sane_lists",   # 목록 파싱 개선
    "toc",          # 목차 앵커 자동 생성
]


def _markdown_to_html(md_text: str, report_type: str) -> str:
    """마크다운을 완전한 HTML 문서로 변환합니다."""
    body_html = markdown.markdown(md_text, extensions=_MD_EXTENSIONS)
    body_html = _wrap_summary_section(body_html)

    today = date.today().strftime("%Y년 %m월 %d일")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{report_type} — {today}</title>
</head>
<body>
{body_html}
</body>
</html>"""


# ──────────────────────────────────────────
# 공개 인터페이스
# ──────────────────────────────────────────

def render_pdf(markdown_text: str, output_path: str, report_type: str = "투자 보고서") -> str:
    """
    마크다운 문자열을 전문 보고서 스타일 PDF로 변환하여 저장합니다.

    Args:
        markdown_text: LLM이 생성한 마크다운 보고서 문자열
        output_path:   저장할 PDF 파일 경로 (예: "/path/to/report.pdf")
        report_type:   보고서 유형 문자열 — HTML <title> 및 로그에 사용

    Returns:
        저장된 PDF 파일 경로
    """
    html_content = _markdown_to_html(markdown_text, report_type)
    HTML(string=html_content).write_pdf(output_path, stylesheets=[_REPORT_CSS])
    print(f"[ReportWriter] PDF 저장 완료: {output_path}")
    return output_path

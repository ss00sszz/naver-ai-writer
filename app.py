import concurrent.futures
import csv
import io
import json
import re
from urllib.parse import urlparse, parse_qs, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="네이버 AI 원고 스튜디오", page_icon="✍️", layout="wide")

MODELS = {
    "초저가 · GPT-5 nano": "gpt-5-nano",
    "경제형 · GPT-5.4 nano": "gpt-5.4-nano",
    "균형형 · GPT-5.4 mini (추천)": "gpt-5.4-mini",
    "고급형 · GPT-5.4": "gpt-5.4",
    "고급형 · GPT-5.5": "gpt-5.5",
    "차세대 경제형 · GPT-6 Luna": "gpt-6-luna",
    "차세대 균형형 · GPT-6.1 Terra": "gpt-6.1-terra",
    "차세대 고급형 · GPT-6.1 Sol": "gpt-6.1-sol",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36", "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"}


def clean(s):
    return re.sub(r"[ \t]+", " ", re.sub(r"\r", "", s or "")).strip()


def article_url(raw):
    raw = raw.strip()
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("https", "http") or not (host == "naver.com" or host.endswith(".naver.com")):
        raise ValueError("네이버 블로그/카페의 http(s) 주소만 지원합니다.")
    if host == "blog.naver.com":
        chunks = [c for c in parsed.path.split("/") if c]
        if len(chunks) >= 2 and chunks[1].isdigit():
            return f"https://m.blog.naver.com/{chunks[0]}/{chunks[1]}"
        q = parse_qs(parsed.query)
        if q.get("blogId") and q.get("logNo"):
            return f"https://m.blog.naver.com/{q['blogId'][0]}/{q['logNo'][0]}"
    if host in ("m.blog.naver.com", "blog.naver.com", "m.cafe.naver.com", "cafe.naver.com"):
        return raw
    raise ValueError("blog.naver.com 또는 cafe.naver.com 글 주소를 입력하세요.")


def fetch_html(url):
    sess = requests.Session()
    # 외부 서비스 도메인으로 리다이렉트되지 않도록 각 홉 검증
    for _ in range(4):
        url = article_url(url)
        res = sess.get(url, headers=HEADERS, timeout=16, allow_redirects=False)
        if 300 <= res.status_code < 400 and res.headers.get("Location"):
            from urllib.parse import urljoin
            url = urljoin(url, res.headers["Location"])
            continue
        res.raise_for_status()
        if len(res.content) > 5_000_000:
            raise ValueError("문서 크기가 너무 큽니다.")
        return res.text, url
    raise ValueError("페이지 이동 횟수가 많습니다.")


def text_of(element):
    if element is None:
        return ""
    fragment = BeautifulSoup(str(element), "lxml")
    for bad in fragment.select("script, style, iframe, button, .se-oglink, .se-sticker, .se-map, .advertisement, .comment, .u_cbox, .cmt"):
        bad.decompose()
    return clean(fragment.get_text("\n", strip=True))


def parse_page(html, url):
    soup = BeautifulSoup(html, "lxml")
    # 구형 블로그 iframe 메인 페이지
    frame = soup.select_one("iframe#mainFrame")
    if frame:
        from urllib.parse import urljoin
        dest = urljoin(url, frame.get("src", ""))
        if dest != url:
            return {"iframe": dest}
    title = ""
    for selector in ("meta[property='og:title']", "meta[name='twitter:title']", "meta[name='title']"):
        t = soup.select_one(selector)
        if t and t.get("content"):
            title = clean(t["content"])
            break
    if not title:
        for selector in (".se-title-text", ".pcol1", "h1", "title"):
            t = soup.select_one(selector)
            if t:
                title = clean(t.get_text(" ", strip=True))
                break
    candidates = []
    for selector in ("div.se-main-container", "div#postViewArea", "div#post-view", "div.post_ct", ".article_viewer", ".ArticleContentBox", ".article_container", "div#tbody", "article", "main"):
        for elt in soup.select(selector):
            s = text_of(elt)
            if len(s) > 100:
                candidates.append((len(s), s))
    body = max(candidates, default=(0, ""))[1]
    if len(body) < 100:
        # OG description은 전문 대신 단순 미리보기일 수 있어 본문으로 인정하지 않음
        return title, ""
    return title, body[:30000]


def fetch_article(url):
    url = article_url(url)
    html, final = fetch_html(url)
    result = parse_page(html, final)
    if isinstance(result, dict) and result.get("iframe"):
        html, final = fetch_html(result["iframe"])
        result = parse_page(html, final)
    title, body = result if isinstance(result, tuple) else ("", "")
    if len(body) < 100:
        raise ValueError("공개 본문을 자동 추출하지 못했습니다. 로그인·멤버 공개·동적 로딩 제한일 수 있습니다. 아래 '직접 붙여넣기'에 제목과 본문을 입력하세요.")
    return {"url": url, "title": title, "body": body, "source": "URL 자동 추출"}


ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title_suggestions": {"type": "array", "items": {"type": "string"}},
        "title_terms": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {"keyword": {"type": "string"}, "sentence": {"type": "string"}}, "required": ["keyword", "sentence"], "additionalProperties": False}},
        "structure": {"type": "string"},
    },
    "required": ["title_suggestions", "title_terms", "keywords", "evidence", "structure"],
    "additionalProperties": False,
}


def api_call(client, model, instructions, content, structured=False, effort="none"):
    kw = {"model": model, "instructions": instructions, "input": content}
    if effort != "none":
        kw["reasoning"] = {"effort": effort}
    if structured:
        kw["text"] = {"format": {"type": "json_schema", "name": "article_analysis", "schema": ANALYSIS_SCHEMA, "strict": True}}
    response = client.responses.create(**kw)
    if not response.output_text:
        raise ValueError("모델 응답에 텍스트가 없습니다. 모델·추론 강도를 조정하세요.")
    return json.loads(response.output_text) if structured else response.output_text


def analyze(api_key, model, effort, article, main_keyword, count):
    client = OpenAI(api_key=api_key, timeout=120, max_retries=1)
    instructions = """당신은 한국어 네이버 블로그·카페 글의 제목/본문을 분석하는 편집자입니다.
원문 사실을 절대 창작하지 마세요. 제목 후보는 원본과 유사한 검색 의도/형태로 쓰되 문장을 그대로 반복하지 않습니다.
제목에 나온 말뿐 아니라 관련한 단어의 띄어쓰기 변화, 조사, 문맥상 동의 표현을 본문에서 찾아 ‘아파트를 보유’, ‘기존 주택담보대출’, ‘추가로 목돈이 필요’처럼 실제 문장 구간을 바탕으로 의미 있는 한국어 구절을 추출하세요.
키워드는 한두 글자 명사 조각보다 검색 가치가 있는 2~8어절의 구절을 우선합니다. 키워드를 원문에서 문맥상 뒷받침할 수 없는 경우 제외하세요.
evidence의 sentence에는 본문에 있는 짧은 근거 문장을 정확히 인용하고, 해당 문장과 매칭되는 키워드를 넣으세요. 키워드는 20~35개 목표(본문 정보가 적으면 가능한 만큼만). 중복과 형태만 다른 표현은 줄이세요.
입력한 메인키워드는 제목 후보에 자연스럽게 반영하세요. 결과는 한국어 JSON으로만 반환하세요."""
    payload = json.dumps({"main_keyword": main_keyword, "suggestion_count": count, "article": {"title": article["title"], "body": article["body"][:22000]}}, ensure_ascii=False)
    data = api_call(client, model, instructions, payload, structured=True, effort=effort)
    data["title_suggestions"] = data.get("title_suggestions", [])[:count]
    data["keywords"] = list(dict.fromkeys(x.strip() for x in data.get("keywords", []) if x.strip()))
    return data


def keyword_lines(value):
    """편집기 입력을 순서 유지하면서 중복 제거합니다."""
    return list(dict.fromkeys(line.strip() for line in value.splitlines() if line.strip()))


def keyword_exclusions(original, selected):
    selected_normalized = {re.sub(r"\s+", "", x).casefold() for x in selected}
    return [word for word in original if re.sub(r"\s+", "", word).casefold() not in selected_normalized]


def keyword_occurrences(text, excluded):
    """띄어쓰기 차이까지 고려해 제외된 구절이 나타나는지 검사합니다."""
    normalized_text = re.sub(r"\s+", "", text).casefold()
    return [word for word in excluded if re.sub(r"\s+", "", word).casefold() in normalized_text]


def strip_excluded_references(text, excluded):
    """구조/원문 참고자료가 삭제된 키워드를 다시 유도하지 않도록 제거합니다."""
    for phrase in sorted(excluded, key=len, reverse=True):
        if not phrase.strip():
            continue
        pattern = r"\s*".join(re.escape(piece) for piece in phrase.split())
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def commit_keywords(i):
    slot = st.session_state.slots[i]
    slot["saved_keywords"] = keyword_lines(st.session_state.get(f"keywords_{i}", ""))


def write_article(api_key, model, effort, article, analysis, main_keyword, custom_prompt, title, index, use_source):
    client = OpenAI(api_key=api_key, timeout=150, max_retries=1)
    instructions = """당신은 한국어 전문 원고 편집자입니다. 사용자의 맞춤 작성 지침을 우선하여 독립적인 새 글을 만드세요.
원문의 문단과 문장을 복제하거나 밀접하게 바꿔쓰지 말고 분석된 개념과 단어를 토대로 자체 구성하세요.
실제 경험, 상담 승인, 고객 사례, 업체 세부정보, 보장 수익, 수치와 규정 등을 근거 없이 지어내지 마세요.
금융 주제는 정확하지 않은 조건·금리·승인·법규를 단정하지 마세요. 확인되지 않은 사실은 조건이 달라질 수 있다고 표현하세요.
작성 지침의 분량, 말투, 키워드 사용 횟수, 제목, 구조를 가능한 충실히 지키세요.
원고 본문만 출력하세요."""
    excluded = analysis.get("excluded_keywords", [])
    payload = {
        "selected_title": title,
        "main_keyword": main_keyword,
        "keywords_to_use": analysis["keywords"],
        "keywords_never_to_include": excluded,
        "structure_reference": strip_excluded_references(analysis["structure"], excluded),
        "custom_prompt": custom_prompt,
        "draft_number": index,
        "variation_instruction": "앞서 작성한 글을 재사용하지 말고 이 차수에 고유한 구성과 전개 방식을 적용하세요.",
        "keyword_priority": "keywords_to_use만 사용 키워드로 간주하세요. keywords_never_to_include는 원문에 있더라도 사용하지 마세요."
    }
    if use_source:
        payload["source_summary_reference"] = {
            "title": strip_excluded_references(article["title"], excluded),
            "body": strip_excluded_references(article["body"][:12000], excluded),
        }

    request_text = json.dumps(payload, ensure_ascii=False)
    if excluded:
        instructions += "\n삭제한 키워드 목록은 사용 금지입니다. 띄어쓰기만 바꿔서 쓰는 것도 금지하며, 참고 원문이나 사용자 프롬프트보다 이 제외 조건을 우선하세요."
    result = api_call(client, model, instructions, request_text, effort=effort)
    # 자동 검사 후 문제가 있으면 1회 수정 요청합니다. 재시도에도 남으면 경고합니다.
    found = keyword_occurrences(result, excluded)
    if found:
        retry_instructions = instructions + "\n원고에 금지 키워드가 섞이지 않도록 전체 글을 다시 작성하세요. 금지 표현을 한 번도 쓰지 마세요."
        retry_payload = json.dumps({"previous_draft": result, "forbidden_matches": found, "keywords_to_use": analysis["keywords"], "selected_title": title, "custom_prompt": custom_prompt, "instruction": "원고 전체를 다시 작성하고 금지 구절을 완전히 제거하세요."}, ensure_ascii=False)
        result = api_call(client, model, retry_instructions, retry_payload, effort=effort)
        remaining = keyword_occurrences(result, excluded)
        if remaining:
            return "⚠️ 제외한 키워드가 여전히 발견되어 확인이 필요합니다: " + ", ".join(remaining) + "\n\n" + result
    return result



st.title("✍️ 네이버 AI 원고 스튜디오")
st.caption("공개 블로그/카페 글 최대 5개 → 제목 추천 → 문맥형 핵심키워드 → 프롬프트 기반 원고 생성")
if "slots" not in st.session_state:
    st.session_state.slots = [{"article": None, "analysis": None, "drafts": [], "error": ""} for _ in range(5)]

with st.sidebar:
    st.header("API · 모델 설정")
    api_key = st.text_input("OpenAI API 키", type="password", help="키는 세션 내 API 호출에 사용하며 파일에 저장하지 않습니다.")
    model_label = st.selectbox("분석 모델", list(MODELS), index=2)
    draft_label = st.selectbox("원고 작성 모델", list(MODELS), index=2)
    analysis_model, draft_model = MODELS[model_label], MODELS[draft_label]
    custom_model = st.text_input("분석 모델 ID 직접 입력 (선택)", placeholder="예: gpt-5.4-mini")
    custom_draft_model = st.text_input("작성 모델 ID 직접 입력 (선택)")
    analysis_model = custom_model.strip() or analysis_model
    draft_model = custom_draft_model.strip() or draft_model
    effort = st.selectbox("분석 추론 강도", ["none", "low", "medium", "high"], index=0)
    draft_effort = st.selectbox("작성 추론 강도", ["none", "low", "medium", "high"], index=0)
    st.caption("추론 강도는 모델에 따라 지원 범위가 다릅니다. 미지원 오류 시 none 또는 low로 변경하세요.")
    st.info("OpenAI API 사용료는 별도 발생합니다. 키를 공개 웹사이트 이용자에게 노출하지 마세요.")

main_keyword = st.text_input("메인키워드 (전체 작업 공통)", placeholder="예: 아파트추가담보대출")
count = st.slider("링크별 추천 제목 수", 5, 15, 10)
st.subheader("1. 링크 5개 입력 / 원문 직접 붙여넣기")
url_inputs = []
for i in range(5):
    with st.expander(f"글 {i+1}", expanded=(i == 0)):
        url = st.text_input(f"네이버 글 URL #{i+1}", key=f"url{i}", placeholder="https://m.blog.naver.com/... 또는 https://cafe.naver.com/...")
        manual_title = st.text_input(f"직접 입력 제목 #{i+1} (선택)", key=f"title{i}")
        manual_body = st.text_area(f"직접 입력 본문 #{i+1} (자동 추출 불가 시)", key=f"body{i}", height=130)
        url_inputs.append((url, manual_title, manual_body))

c1, c2 = st.columns(2)
with c1:
    fetch_clicked = st.button("① 5개 글 가져오기", type="primary", use_container_width=True)
with c2:
    analyze_clicked = st.button("② 5개 글 AI 분석", use_container_width=True)

if fetch_clicked:
    tasks = []
    for i, (url, title, body) in enumerate(url_inputs):
        st.session_state.slots[i] = {"article": None, "analysis": None, "drafts": [], "error": ""}
        for widget_key in (f"chosen_{i}", f"edited_title_{i}", f"keywords_{i}"):
            st.session_state.pop(widget_key, None)
        st.session_state.slots[i].pop("saved_keywords", None)
        if body.strip():
            st.session_state.slots[i]["article"] = {"url": url, "title": title or f"글 {i+1}", "body": body.strip(), "source": "직접 입력"}
        elif url.strip():
            tasks.append((i, url.strip(), title.strip()))
    if tasks:
        with st.spinner("공개 페이지 읽는 중..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                future_map = {pool.submit(fetch_article, url): (i, title) for i, url, title in tasks}
                for f in concurrent.futures.as_completed(future_map):
                    i, title = future_map[f]
                    try:
                        article = f.result()
                        if title:
                            article["title"] = title
                        st.session_state.slots[i]["article"] = article
                    except Exception as ex:
                        st.session_state.slots[i]["error"] = str(ex)
    st.success("가져오기 작업이 끝났습니다. 아래 글별 상태를 확인하세요.")

if analyze_clicked:
    if not api_key:
        st.error("API 키를 입력하세요.")
    else:
        tasks = [(i, s["article"]) for i, s in enumerate(st.session_state.slots) if s["article"]]
        if not tasks:
            st.warning("먼저 글 가져오기를 진행하거나 제목/본문을 붙여넣으세요.")
        else:
            with st.spinner("GPT로 글별 제목·핵심키워드 분석 중..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    futures = {pool.submit(analyze, api_key, analysis_model, effort, article, main_keyword, count): i for i, article in tasks}
                    for f in concurrent.futures.as_completed(futures):
                        i = futures[f]
                        try:
                            new_analysis = f.result()
                            st.session_state.slots[i]["analysis"] = new_analysis
                            st.session_state.slots[i]["saved_keywords"] = list(new_analysis["keywords"])
                            st.session_state.slots[i]["error"] = ""
                            for widget_key in (f"chosen_{i}", f"edited_title_{i}", f"keywords_{i}"):
                                st.session_state.pop(widget_key, None)
                        except Exception as ex:
                            st.session_state.slots[i]["error"] = "분석 오류: " + str(ex)
            st.success("분석 요청이 끝났습니다.")

st.subheader("2. 분석 결과 / 키워드 수정")
for i, slot in enumerate(st.session_state.slots):
    article, analysis = slot["article"], slot["analysis"]
    if not article and not slot["error"]:
        continue
    with st.expander(f"글 {i+1} · {article['title'] if article else '읽기 실패'}", expanded=bool(analysis)):
        if slot["error"]:
            st.error(slot["error"])
        if article:
            st.caption(f"출처: {article['source']} · 본문 {len(article['body']):,}자")
            st.text_area("추출 제목", value=article["title"], key=f"source_title_{i}", disabled=True)
            st.text_area("추출 본문 미리보기", value=article["body"][:5000], height=130, key=f"preview_{i}", disabled=True)
        if analysis:
            st.markdown("**추천 제목**")
            options = analysis["title_suggestions"]
            for j, t in enumerate(options):
                st.write(f"{j+1}. {t}")
            if options:
                # 제목 선택 시에만 편집란을 덮어씁니다. 그 외 재실행에서는 수정 내용을 유지합니다.
                chosen_key = f"chosen_{i}"
                edited_key = f"edited_title_{i}"
                if chosen_key not in st.session_state or st.session_state[chosen_key] not in options:
                    st.session_state[chosen_key] = options[0]
                if edited_key not in st.session_state:
                    st.session_state[edited_key] = st.session_state[chosen_key]

                def apply_selected_title(selection_key=chosen_key, target_key=edited_key):
                    st.session_state[target_key] = st.session_state[selection_key]

                st.selectbox("원고에 사용할 제목", options, key=chosen_key, on_change=apply_selected_title)
                st.text_area(
                    "원고 제목 직접 수정 (선택)",
                    key=edited_key,
                    height=90,
                    help="추천 제목을 선택하면 자동 입력됩니다. 수정할 경우 Ctrl+Enter로 입력을 확정하세요. 수정하지 않으면 선택한 추천 제목으로 작성합니다.",
                )
                st.caption("제목을 수정하지 않아도 됩니다. 수정한 경우 원고 생성 시 해당 제목이 적용됩니다.")
            st.caption("제목에서 포착한 핵심 개념: " + ", ".join(analysis.get("title_terms", [])))
            if "saved_keywords" not in slot:
                slot["saved_keywords"] = list(analysis["keywords"])
            if f"keywords_{i}" not in st.session_state:
                st.session_state[f"keywords_{i}"] = "\n".join(slot["saved_keywords"])
            st.text_area(
                "핵심키워드 수정 (한 줄에 하나)",
                height=220,
                key=f"keywords_{i}",
                help="삭제 또는 추가한 후 아래 '키워드 수정 저장' 버튼을 누르세요. Ctrl+Enter로 입력을 확정해도 됩니다.",
                on_change=commit_keywords,
                args=(i,),
            )
            if st.button("💾 키워드 수정 저장", key=f"save_keywords_{i}"):
                commit_keywords(i)
                st.success("수정 키워드가 저장되었습니다. 생성 시 이 목록만 사용합니다.")
            saved = slot["saved_keywords"]
            removed = keyword_exclusions(analysis["keywords"], saved)
            st.caption(f"저장된 키워드: {len(saved)}개 · 삭제하여 제외된 키워드: {len(removed)}개")
            if removed:
                st.info("원고에서 제외할 키워드: " + ", ".join(removed))
            if keyword_lines(st.session_state.get(f"keywords_{i}", "")) != saved:
                st.warning("편집 내용이 아직 저장되지 않았습니다. '키워드 수정 저장'을 클릭하세요.")
            ev = analysis.get("evidence", [])
            if ev:
                with st.expander("본문 근거 문장 보기"):
                    st.dataframe(pd.DataFrame(ev), use_container_width=True, hide_index=True)
            st.caption("원문 구조 분석: " + analysis.get("structure", ""))

st.subheader("3. 맞춤 프롬프트로 원고 생성")
custom_prompt = st.text_area("원고 작성 프롬프트 (모든 글 공통)", height=190, value="1인칭 존댓말로 작성해 주세요.\n메인키워드를 제목과 본문에 자연스럽게 사용해 주세요.\n입력한 핵심키워드를 문맥에 맞춰 활용해 주세요.\n중복 문장 없이 새 구성으로 원고를 작성해 주세요.\n사실로 확인되지 않은 수치나 경험을 지어내지 마세요.")
num_drafts = st.number_input("글당 생성 원고 수", min_value=1, max_value=5, value=1, step=1)
use_source = st.checkbox("작성 시 원본 본문도 참고 (기본은 분석 키워드와 구조만 활용)", value=False)
if st.button("③ 분석된 글 모두 원고 생성", type="primary", use_container_width=True):
    if not api_key:
        st.error("API 키를 입력하세요.")
    else:
        tasks = []
        for i, slot in enumerate(st.session_state.slots):
            if not slot["analysis"] or not slot["article"]:
                continue
            analysis = dict(slot["analysis"])
            # 편집 화면에 남은 문자가 있다면 마지막 확정값을 사용합니다.
            selected = slot.get("saved_keywords", list(analysis["keywords"]))
            analysis["keywords"] = list(selected)
            analysis["excluded_keywords"] = keyword_exclusions(slot["analysis"]["keywords"], selected)
            title = (st.session_state.get(f"edited_title_{i}") or "").strip() or st.session_state.get(f"chosen_{i}") or slot["article"]["title"]
            conflicts = keyword_occurrences(title + "\n" + main_keyword, analysis["excluded_keywords"])
            if conflicts:
                st.warning(f"글 {i+1}: 삭제한 키워드가 제목 또는 메인키워드에도 포함되어 있습니다: {', '.join(conflicts)}. 제목/메인키워드를 확인하세요.")
            slot["drafts"] = []
            for n in range(int(num_drafts)):
                tasks.append((i, n + 1, slot["article"], analysis, title))
        if not tasks:
            st.warning("분석된 글이 없습니다. 먼저 ①과 ②를 실행하세요.")
        else:
            with st.spinner(f"원고 {len(tasks)}개 작성 중..."):
                # 과도한 동시 호출 방지. 최대 5개 병렬 처리
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    futures = {pool.submit(write_article, api_key, draft_model, draft_effort, article, anal, main_keyword, custom_prompt, title, n, use_source): (i, n, title) for i, n, article, anal, title in tasks}
                    for f in concurrent.futures.as_completed(futures):
                        i, n, title = futures[f]
                        try:
                            st.session_state.slots[i]["drafts"].append({"number": n, "title": title, "text": f.result()})
                        except Exception as ex:
                            st.session_state.slots[i]["drafts"].append({"number": n, "title": title, "text": "[생성 실패] " + str(ex)})
            st.success("원고 생성 요청이 끝났습니다.")

all_drafts = []
for i, slot in enumerate(st.session_state.slots):
    for draft in sorted(slot["drafts"], key=lambda d: d["number"]):
        all_drafts.append({"글번호": i + 1, "차수": draft["number"], "제목": draft["title"], "원고": draft["text"]})
        with st.expander(f"원고 {i+1}-{draft['number']} · {draft['title']}"):
            st.text_area("원고", draft["text"], height=380, key=f"draft_{i}_{draft['number']}")
            st.download_button("이 원고 TXT 저장", (draft["title"] + "\n\n" + draft["text"]).encode("utf-8-sig"), file_name=f"draft_{i+1}_{draft['number']}.txt", key=f"dl_{i}_{draft['number']}")
if all_drafts:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["글번호", "차수", "제목", "원고"])
    writer.writeheader()
    writer.writerows(all_drafts)
    st.download_button("📥 모든 원고 CSV 다운로드", buf.getvalue().encode("utf-8-sig"), "naver_ai_drafts.csv", "text/csv", use_container_width=True)

st.divider()
st.caption("주의: 네이버에서 공개되었더라도 자동 수집이 차단되는 글이 있습니다. 로그인/멤버 전용·비공개 글 접근을 우회하지 않습니다. 본문 추출 실패 시 직접 붙여넣기를 사용하세요. 원문 저작권 및 네이버 이용정책을 준수하고 게시 전 사실관계를 검토하세요.")

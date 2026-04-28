import re
import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SEARCH_HOME = "https://zrzy.jiangsu.gov.cn/elsearch/search/index?areaCode=320000"
KEYWORD = "城乡规划编制单位乙级资质"
OUT_CSV = "江苏乙级资质公司名单.csv"


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text).strip()


def extract_company_name(title: str) -> str:
    """
    从标题中提取公司名称
    """
    title = clean_text(title)

    patterns = [
        r"关于(.*?)申请城乡规划编制单位乙级资质",
        r"关于(.*?)城乡规划编制单位乙级资质",
        r"^(.*?)城乡规划编制单位乙级资质",
    ]

    for pat in patterns:
        m = re.search(pat, title)
        if m:
            name = m.group(1).strip("《》“”\"'：:，,。. ")
            return name

    return ""


def do_search(page):
    """
    打开搜索页，输入关键词并点击搜索
    """
    page.goto(SEARCH_HOME, wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    # 先尝试常见输入框
    input_candidates = [
        "input[type='text']",
        "input[placeholder*='搜索']",
        "input[name='content']",
        "input",
    ]

    search_input = None
    for sel in input_candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            for i in range(loc.count()):
                try:
                    item = loc.nth(i)
                    if item.is_visible():
                        search_input = item
                        break
                except:
                    pass
        if search_input:
            break

    if search_input is None:
        raise Exception("没有找到搜索输入框")

    search_input.fill(KEYWORD)
    time.sleep(1)

    # 尝试点击搜索按钮
    button_candidates = [
        "button:has-text('搜索')",
        "input[type='submit']",
        "text=搜索",
        ".search-btn",
        ".btn-search",
        "button",
    ]

    clicked = False
    for sel in button_candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            for i in range(loc.count()):
                try:
                    btn = loc.nth(i)
                    if btn.is_visible():
                        txt = clean_text(btn.inner_text() or "")
                        # button 选择器最后兜底，允许无文字
                        if ("搜索" in txt) or (sel == "button"):
                            btn.click()
                            clicked = True
                            break
                except:
                    pass
        if clicked:
            break

    if not clicked:
        # 如果按钮实在找不到，就直接回车
        search_input.press("Enter")

    page.wait_for_load_state("networkidle")
    time.sleep(2)


def parse_current_page(page, page_no: int):
    page.wait_for_selector(".item-title", timeout=15000)
    items = page.locator(".item-title")
    count = items.count()

    print(f"第 {page_no} 页，共检测到 {count} 条结果")

    results = []
    for i in range(count):
        try:
            item = items.nth(i)
            raw_title = clean_text(item.inner_text())
            company_name = extract_company_name(raw_title)

            results.append({
                "page_no": page_no,
                "raw_title": raw_title,
                "company_name": company_name,
            })

            print(f"[第{page_no}页 第{i+1}条] 公司名: {company_name} | 标题: {raw_title}")

        except Exception as e:
            print(f"[第{page_no}页 第{i+1}条] 解析失败: {e}")

    return results


def goto_next_page(page):
    """
    点击下一页
    """
    selectors = [
        "a:has-text('下一页')",
        "text=下一页",
        "a:has-text('下页')",
        ".layui-laypage-next",
        ".pagination a:has-text('下一页')",
        ".page a:has-text('下一页')",
    ]

    for sel in selectors:
        loc = page.locator(sel)
        if loc.count() > 0:
            for i in range(loc.count()):
                try:
                    btn = loc.nth(i)
                    if not btn.is_visible():
                        continue

                    cls = (btn.get_attribute("class") or "").lower()
                    if "disabled" in cls or "layui-disabled" in cls:
                        return False

                    btn.click()
                    page.wait_for_load_state("networkidle")
                    time.sleep(2)
                    return True
                except:
                    pass

    return False


def main():
    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(20000)

        do_search(page)

        page_no = 1
        seen_titles = set()

        while True:
            try:
                rows = parse_current_page(page, page_no)

                if not rows:
                    print("当前页没有结果，停止。")
                    break

                first_title = rows[0]["raw_title"]
                if first_title in seen_titles:
                    print("检测到重复页面，停止翻页。")
                    break
                seen_titles.add(first_title)

                all_rows.extend(rows)

                if not goto_next_page(page):
                    print("没有下一页了。")
                    break

                page_no += 1

            except PWTimeout:
                print(f"第 {page_no} 页加载超时，停止。")
                break
            except Exception as e:
                print(f"运行出错：{e}")
                break

        browser.close()

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("没有抓到数据。")
        return

    df = df[df["company_name"].astype(str).str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["company_name", "raw_title"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n抓取完成，共 {len(df)} 条")
    print(f"结果已保存到：{OUT_CSV}")


if __name__ == "__main__":
    main()

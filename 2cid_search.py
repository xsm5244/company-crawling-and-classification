import csv
import re
from pathlib import Path
from time import sleep
from typing import Optional, List, Dict

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ================= 可配置参数 =================
CSV_PATH = Path(r"E:\BaiduSyncdisk\企业科研\yiji2026jiangsu\1名称爬取\江苏乙级资质公司名单(全).csv")
OUT_PATH = CSV_PATH.with_name("newcode_yijiplus.csv")

BATCH_SIZE = 50
MAX_RETRY = 2
PAGE_TIMEOUT = 20000
START_INDEX = 0

HOME_URL = "https://data.ggzy.gov.cn/#/home"
# ============================================


def extract_credit_code_from_url(url: str) -> Optional[str]:
    m = re.search(r"[?#&]id=([0-9A-Z]{18})(?:[&#]|$)", url)
    return m.group(1) if m else None


def try_read_csv_with_encodings(csv_file: Path):
    encodings = ["utf-8-sig", "gb18030", "utf-8", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            with csv_file.open("r", newline="", encoding=enc, errors="ignore") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)
                print(f"成功使用编码读取: {enc}")
                print(f"检测到表头: {fieldnames}")
                return fieldnames, rows
        except Exception as e:
            last_error = e

    raise last_error


def read_csv_safe(csv_file: Path) -> List[Dict[str, str]]:
    fieldnames, raw_rows = try_read_csv_with_encodings(csv_file)

    if not fieldnames or "company_name" not in fieldnames:
        print("未检测到 company_name 列，实际表头为：", fieldnames)
        return []

    rows = []
    for row in raw_rows:
        company_name = (row.get("company_name") or "").strip()
        if company_name:
            rows.append({
                "company_name": company_name,
                "统一社会信用代码": ""
            })
    return rows


def write_batch(rows: List[Dict[str, str]], out_path: Path) -> None:
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["company_name", "统一社会信用代码"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def query_credit_code(page, name: str, debug_index: int = 0) -> Optional[str]:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"\n开始查询：{name}（第 {attempt} 次尝试）")
            page.goto(HOME_URL, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
            page.wait_for_timeout(3000)

            print("当前首页 URL：", page.url)
            print("页面标题：", page.title())

            # 把当前页所有 input 打印出来，方便看真实搜索框
            inputs = page.locator("input")
            input_count = inputs.count()
            print("input 数量：", input_count)

            target = None
            for i in range(input_count):
                try:
                    item = inputs.nth(i)
                    placeholder = item.get_attribute("placeholder")
                    input_type = item.get_attribute("type")
                    print(f"input[{i}] placeholder={placeholder!r}, type={input_type!r}")
                    if target is None and (
                        (placeholder and ("搜索" in placeholder or "企业" in placeholder or "单位" in placeholder or "公司" in placeholder))
                        or input_type == "text"
                    ):
                        target = item
                except Exception:
                    pass

            if target is None and input_count > 0:
                target = inputs.nth(0)

            if target is None:
                page.screenshot(path=f"debug_no_input_{debug_index}.png", full_page=True)
                raise Exception("未找到输入框")

            target.click()
            page.wait_for_timeout(500)
            target.fill("")
            page.wait_for_timeout(500)
            target.fill(name)
            page.wait_for_timeout(1500)

            print("输入后当前 URL：", page.url)

            # 先按回车
            target.press("Enter")
            print("已按 Enter，等待跳转...")
            page.wait_for_timeout(5000)
            print("Enter 后 URL：", page.url)

            code = extract_credit_code_from_url(page.url)
            if code:
                return code

            # 再找“搜索”相关按钮
            clickable_candidates = [
                "text=搜索",
                "button",
                "[role='button']",
                "span",
                "div"
            ]

            clicked = False
            for sel in clickable_candidates:
                try:
                    loc = page.locator(sel)
                    count = min(loc.count(), 20)
                    for i in range(count):
                        node = loc.nth(i)
                        try:
                            txt = (node.inner_text() or "").strip()
                        except Exception:
                            txt = ""
                        if "搜索" in txt:
                            print(f"尝试点击包含‘搜索’的元素：selector={sel}, index={i}, text={txt!r}")
                            node.click(timeout=2000)
                            clicked = True
                            page.wait_for_timeout(5000)
                            print("点击后 URL：", page.url)
                            code = extract_credit_code_from_url(page.url)
                            if code:
                                return code
                            break
                    if clicked:
                        break
                except Exception:
                    pass

            # 看看是否出现 company 链接
            links = page.locator("a")
            link_count = min(links.count(), 30)
            print("a 标签数量：", link_count)
            for i in range(link_count):
                try:
                    href = links.nth(i).get_attribute("href")
                    text = (links.nth(i).inner_text() or "").strip()
                    if href:
                        print(f"a[{i}] href={href!r}, text={text!r}")
                    if href and "#/company?id=" in href:
                        print("找到 company 链接，点击进入")
                        links.nth(i).click(timeout=2000)
                        page.wait_for_timeout(5000)
                        print("点击 company 链接后 URL：", page.url)
                        code = extract_credit_code_from_url(page.url)
                        if code:
                            return code
                except Exception:
                    pass

            # 最后截图留证据
            screenshot_path = f"debug_search_fail_{debug_index}.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print("未成功跳转，已保存截图：", screenshot_path)
            return None

        except PWTimeout:
            print("超时")
            if attempt == MAX_RETRY:
                return None
            sleep(3)
        except Exception as e:
            print("异常：", repr(e))
            if attempt == MAX_RETRY:
                screenshot_path = f"debug_exception_{debug_index}.png"
                try:
                    page.screenshot(path=screenshot_path, full_page=True)
                    print("异常截图已保存：", screenshot_path)
                except Exception:
                    pass
                return None
            sleep(3)

    return None


def main() -> None:
    print("CSV_PATH =", CSV_PATH)
    print("OUT_PATH =", OUT_PATH)
    print("START_INDEX =", START_INDEX)

    all_rows = read_csv_safe(CSV_PATH)
    total = len(all_rows)
    print("总有效公司数 =", total)

    if START_INDEX >= total:
        print("起始索引超出范围，无数据可处理。")
        return

    rows = all_rows[START_INDEX:]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context()
        page = context.new_page()

        try:
            for batch_start in range(0, len(rows), BATCH_SIZE):
                batch = rows[batch_start: batch_start + BATCH_SIZE]

                print(
                    f"\n处理第 {START_INDEX + batch_start + 1} ~ "
                    f"{START_INDEX + batch_start + len(batch)} 条"
                )

                for i, row in enumerate(batch, start=1):
                    name = row["company_name"]
                    code = query_credit_code(page, name, debug_index=START_INDEX + batch_start + i)
                    row["统一社会信用代码"] = code or "未查到"
                    print(f"[{i}/{len(batch)}] {name} -> {row['统一社会信用代码']}")

                    # 先只跑前 1 条调试
                    return

                write_batch(batch, OUT_PATH)

        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()

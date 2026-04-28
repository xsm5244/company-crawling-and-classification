from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
import os
from datetime import datetime

LIST_PATH = r'/Users/mac/Desktop/tongbu/企业科研/yiji2026jiangsu/1名称爬取/yijiplus.csv'
OUTPUT_CSV = r'/Users/mac/Desktop/tongbu/企业科研/yiji2026jiangsu/1名称爬取/2cid_inf.csv'

company_list = pd.read_csv(LIST_PATH, dtype=str).fillna('').to_dict('records')


def make_driver():
    opt = Options()
    opt.add_argument('--headless=new')
    opt.add_argument('--disable-gpu')
    opt.add_argument('--window-size=1920,1080')
    opt.add_argument('--no-sandbox')
    opt.add_argument('--disable-dev-shm-usage')
    opt.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    driver = webdriver.Chrome(options=opt)
    driver.set_page_load_timeout(40)
    return driver


def wait_project_table(driver, timeout=20):
    """
    等待项目表格出现
    """
    end_time = time.time() + timeout
    selectors = [
        '.ivu-table-body tbody tr',
        '.ivu-table-wrapper tbody tr',
        'tbody tr'
    ]

    while time.time() < end_time:
        for sel in selectors:
            try:
                rows = driver.find_elements(By.CSS_SELECTOR, sel)
                if rows:
                    for r in rows:
                        tds = r.find_elements(By.TAG_NAME, 'td')
                        if len(tds) >= 2:
                            return sel
            except:
                pass
        time.sleep(1)

    raise TimeoutException("未等到项目表格出现")


def extract_rows_from_current_page(driver, name, cid):
    rows_buffer = []

    row_selectors = [
        '.ivu-table-body tbody tr',
        '.ivu-table-wrapper tbody tr',
        'tbody tr'
    ]

    trs = []
    for sel in row_selectors:
        trs = driver.find_elements(By.CSS_SELECTOR, sel)
        if trs:
            break

    for tr in trs:
        tds = [td.text.strip() for td in tr.find_elements(By.TAG_NAME, 'td')]

        if not tds or len(tds) < 6:
            continue

        if tds[0] in ['序号', '项目名称']:
            continue

        rows_buffer.append({
            'company_name': name,
            'cid': cid,
            'seq': tds[0],
            'tenderee': tds[1],
            'project_name': tds[2],
            'project_type': tds[3],
            'pub_date': tds[4],
            'bid_amount': tds[5],
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    return rows_buffer


def goto_next_page(driver):
    """
    翻到下一页
    """
    selectors = [
        '.ivu-page-next:not(.ivu-page-disabled)',
        '.ivu-page-next'
    ]

    for sel in selectors:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            if not btns:
                continue

            btn = btns[0]
            cls = btn.get_attribute("class") or ""
            if "ivu-page-disabled" in cls:
                return False

            driver.execute_script("arguments[0].click();", btn)
            time.sleep(2)
            return True
        except:
            pass

    return False


for idx, item in enumerate(company_list, 1):
    name = item.get('name', '').strip()
    cid = item.get('统一社会信用代码', '').strip()

    if not cid or cid == '未查到':
        print(f'[{idx}] {name} 跳过')
        continue

    # 直接进入详情页
    url = f'https://data.ggzy.gov.cn/#/company?id={cid}'
    print(f'[{idx}] {name} 开始...')
    print(f'    URL: {url}')

    driver = None
    all_rows = []

    try:
        driver = make_driver()
        driver.get(url)
        time.sleep(4)

        used_selector = wait_project_table(driver, timeout=20)
        print(f'    命中表格选择器: {used_selector}')

        while True:
            page_rows = extract_rows_from_current_page(driver, name, cid)
            all_rows.extend(page_rows)

            moved = goto_next_page(driver)
            if not moved:
                break

        if all_rows:
            df_new = pd.DataFrame(all_rows)
            header = not os.path.exists(OUTPUT_CSV)
            df_new.to_csv(
                OUTPUT_CSV,
                mode='a',
                index=False,
                header=header,
                encoding='utf-8-sig'
            )
            print(f'✅ {name} 追加 {len(all_rows)} 条')
        else:
            print(f'⚠️ {name} 已进入详情页，但未抓到表格数据')

    except Exception as e:
        print(f'❌ {name} 异常：{repr(e)}')
        if driver is not None:
            try:
                driver.save_screenshot(f'/Users/mac/Desktop/{idx}_debug.png')
                print(f'    已保存截图: /Users/mac/Desktop/{idx}_debug.png')
            except:
                pass

    finally:
        if driver is not None:
            try:
                driver.quit()
            except:
                pass

print('🎉 全部完成，文件：', OUTPUT_CSV)

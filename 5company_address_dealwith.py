import pandas as pd
import re
import os
from collections import defaultdict

# ========= 1. 输入输出路径 =========
addr_file = r'E:\BaiduSyncdisk\企业科研\yiji2026jiangsu\3对地址的提取\company_real_addresses.csv'
div4_file = r'E:\BaiduSyncdisk\企业科研\yiji2026jiangsu\3对地址的提取\四级表_去市地区区_保留2+.csv'

out_file = r'E:\BaiduSyncdisk\企业科研\yiji2026jiangsu\3对地址的提取\company_real_addresses_加市级.csv'
fail_file = r'E:\BaiduSyncdisk\企业科研\yiji2026jiangsu\3对地址的提取\company_real_addresses_未识别市级.csv'

# ========= 2. 读取数据 =========
df = pd.read_csv(addr_file, dtype=str).fillna('')
div4 = pd.read_csv(div4_file, dtype=str).fillna('')

# ========= 3. 清洗四级表 =========
for col in ['省份', '城市', '区县', '街道']:
    if col not in div4.columns:
        raise ValueError(f'四级表缺少列: {col}')
    div4[col] = div4[col].astype(str).str.strip()

# 去掉空行
div4 = div4[
    (div4['城市'] != '') |
    (div4['区县'] != '') |
    (div4['街道'] != '')
].copy()

# ========= 4. 构造映射 =========
# 4.1 区县 -> 可能对应的城市集合
dist_to_cities = defaultdict(set)
for _, row in div4.iterrows():
    dist = row['区县']
    city = row['城市']
    if dist and city:
        dist_to_cities[dist].add(city)

# 4.2 街道 -> 可能对应的区县集合
street_to_dists = defaultdict(set)
for _, row in div4.iterrows():
    street = row['街道']
    dist = row['区县']
    if street and dist:
        street_to_dists[street].add(dist)

# ========= 5. 构造正则池 =========
def make_pattern(values):
    vals = sorted({v.strip() for v in values if str(v).strip()}, key=len, reverse=True)
    vals = [re.escape(v) for v in vals]
    return '|'.join(vals)

city_re = make_pattern(div4['城市'].tolist())
dist_re = make_pattern(div4['区县'].tolist())
street_re = make_pattern(div4['街道'].tolist())

# ========= 6. 提取市级 =========
def extract_city(address: str):
    if pd.isna(address) or str(address).strip() == '':
        return None

    addr = str(address).strip()

    # 1) 直接匹配城市
    m_city = re.search(city_re, addr) if city_re else None
    if m_city:
        return m_city.group()

    # 2) 匹配区县 -> 反查城市
    m_dist = re.search(dist_re, addr) if dist_re else None
    if m_dist:
        dist = m_dist.group()
        cities = list(dist_to_cities.get(dist, []))
        if len(cities) == 1:
            return cities[0]
        elif len(cities) > 1:
            # 如果一个区县对应多个城市，优先看地址里有没有这些城市名残片
            for c in cities:
                if c in addr:
                    return c
            return cities[0]

    # 3) 匹配街道 -> 反查区县 -> 再反查城市
    m_street = re.search(street_re, addr) if street_re else None
    if m_street:
        street = m_street.group()
        dists = list(street_to_dists.get(street, []))
        if len(dists) == 1:
            dist = dists[0]
            cities = list(dist_to_cities.get(dist, []))
            if len(cities) == 1:
                return cities[0]
            elif len(cities) > 1:
                for c in cities:
                    if c in addr:
                        return c
                return cities[0]
        elif len(dists) > 1:
            # 多个区县时，优先看地址里是否包含区县名
            for dist in dists:
                if dist in addr:
                    cities = list(dist_to_cities.get(dist, []))
                    if len(cities) == 1:
                        return cities[0]
                    elif len(cities) > 1:
                        for c in cities:
                            if c in addr:
                                return c
                        return cities[0]
            # 如果地址里没有区县名，就取第一个区县再反查
            dist = dists[0]
            cities = list(dist_to_cities.get(dist, []))
            if cities:
                return cities[0]

    return None

# ========= 7. 应用到地址表 =========
if 'address' not in df.columns:
    raise ValueError('地址表缺少 address 列')

df['市级'] = df['address'].apply(extract_city)

# ========= 8. 导出 =========
out_dir = os.path.dirname(out_file)
os.makedirs(out_dir, exist_ok=True)

df.to_csv(out_file, index=False, encoding='utf-8-sig')

df_fail = df[df['市级'].isna() | (df['市级'] == '')].copy()
df_fail.to_csv(fail_file, index=False, encoding='utf-8-sig')

print('处理完成')
print('输出文件：', out_file)
print('未识别文件：', fail_file)
print('总条数：', len(df))
print('识别到市级条数：', (df['市级'].notna() & (df['市级'] != '')).sum())
print('未识别条数：', len(df_fail))

# ========= 9. 打印前几条看看 =========
print('\n前10条结果预览：')
print(df[['name', 'cid', 'address', '市级']].head(10).to_string(index=False))

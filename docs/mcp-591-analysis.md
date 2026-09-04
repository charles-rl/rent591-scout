# mcp-591 Analysis (`asgard-ai-platform/mcp-591`)

Vendored at `external/mcp-591/` (MIT). Pure `requests` client; no package install needed.
Source of truth for implementation: `mcp_591/client.py`, `mcp_591/constants.py`, `mcp_591/server.py`.

## Auth & Session Setup (`Client591.__init__`)

- Mobile UA: `Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) …Chrome/147.0.0.0 Mobile Safari/537.36`
- Headers: `device=touch`, `deviceid=<uuid4().hex>`, `origin=https://m.591.com.tw`, `referer=https://m.591.com.tw/`
- Cookie: `T591_TOKEN=<device_id>`
- `verify=False` (591 cert lacks Subject Key Identifier)
- No login required. `timestamp` param = `int(time.time()*1000)`.

## Endpoints

| Method | URL | Notes |
|---|---|---|
| `search_sale` | `GET https://bff-house.591.com.tw/v1/touch/sale/list` | params: `type=sale, version=2017, regionid, sectionidStr(csv), firstRow, newPageSize, device, device_id, timestamp, price_str, kind, shape_str, pattern_str, toilet_str, area_str, age_str, keywords` |
| `get_sale_detail` | `GET https://bff-house.591.com.tw/v1/touch/sale/detail` | params: `id=S{post_id}, device, device_id` |
| `search_rent` | `GET https://bff-house.591.com.tw/v3/web/rent/list` | params: `regionid, sectionid(csv), firstRow, timestamp, kind, shape(csv), pattern(csv), rentprice, keywords` |
| `get_rent_detail` | `GET https://bff-house.591.com.tw/v2/web/rent/detail` | params: `id={post_id}` |

### Rent search (returns `data.items[]`)
- `data.total` (string), `data.firstRow` (pagination offset)

### Rent detail (returns `data{}`)
- Rich nested payload. `server.py` extracts a subset; **we call `Client591` directly and store the full JSON**.

## Rent search item — full key inventory
`id, type, kind, kind_name, title, photoList[] (510x400 thumb URLs), url, preferred, price (str), price_unit, price_has_carport, price_per, price_per_unit, floor_name, area_name, layoutStr, fitment_name, community_name, community_id, address, tags[], refresh_tag_visible, refresh_time, browse_count, role_name, good_house, is_video, is_ai_video, ai_title_id, show_ai_title, labels[], surrounding{type,desc,distance}, land_type, ding_kind_alias_name, ding_kind_name, ding_rent_price, ground_type[], sectionid, regionid, area (float 坪), mvip, social_house, cover, video{id,is_video,video_type,source,video_rel_id,cust_video_cover,file_id,video_url(m3u8),video_pic,cover,video_play_count,normal{},ai[]}`

## Rent detail — full key inventory (`data{}`)
`version, title, status (open/closed), deposit, kind, relieved, regionId, sectionId, shareInfo{}, meta{title,keywords,description,ogimage}, headInfo, navData[], tags[{id,value}], price (str), priceUnit, containCost[], info[{name,value,key}], publish{postTime,updateTime}, address{data,value,lat,lng,traffic,station,distance}, houseInfo{data:[{name,value,key,cate,help,alias}]}, service{facility:[{key,active,name}],notice:[{key,name}]}, preference{}, remark{content}, surround{title,address,lat,lng,data:[{name,key,children:[{type,name,distance,distanceTxt}]}]}, cost{data:[{name,value,key}]}, information{data[]}, linkInfo{name,roleName,mobile,phone,avatar,chargeTxt,certificateStatus,isServiceFee,isrecmoney,line,uid,…}, favData{}, positionRound{communityName,communityId,lat,lng,mapData[]}, gtm_detail_data{item_id,item_name,region_name,section_name,kind_name,price_name,shape_name,layout_name,area_name,floor_name,facility_name,label_name,…}`

### Key facts
- `houseInfo.data` keys: `leaseTime, comeDate, degree, purpose, buildArea, pet, cook, hasCertificate` (+ others).
- `service.facility` keys: `fridge, washer, tv, cold, heater, bed, closet, fourth, net, gas, sofa, table_chairs, balcony, lift, park`.
- `info` keys: `layout, area, floor, shape`.
- **Images are NOT in rent_detail** — `photoList` lives in the search item; `meta.ogimage`/`favData.thumb` are fallbacks. `photoList` URLs carry `!510x400.jpg` resize suffix → **strip suffix (`!…`) to fetch original resolution**.
- `get_rent_detail` returns `{}` (empty data) for delisted IDs → treat as dead.

## Constants (`mcp_591/constants.py`)
- `REGIONS` (id→name, 22 counties incl. 台北市=1, 新北市=3, 桃園市=6…), `SECTIONS` (id→(name, region_id)), `SECTIONS_BY_REGION`.
- `RENT_KINDS`: 1=整層住家, 2=獨立套房, 3=分租套房, 4=雅房, 8=車位.
- `SHAPES`: 1=公寓, 2=電梯大樓, 3=透天厝, 4=別墅. `PATTERNS`: 1..5房. `TOILETS`, `AREAS`, `AGES`.

## Behavior notes
- Rate limiting / anti-bot: undocumented API; README warns against bulk/high-frequency use. On failure `raise_for_status` propagates.
- `server.py` `_filter_*` drops most fields → **bypass server layer, use `Client591` directly** for zero data loss.
- Python 3.14 claim in README is advisory; `client.py` + `constants.py` run fine on 3.12 (plain requests/data) → vendored as `src/client591.py` + `src/constants591.py`.

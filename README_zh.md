<div align="center">
  <h1>SciNet: 闈㈠悜鑷姩鍖栫瀛︾爺绌剁殑澶ц妯＄煡璇嗗浘璋?/h1>
</div>

<p align="center">
  涓€涓彲閫氳繃 pip 瀹夎鐨?SciNet 瀹㈡埛绔笌鍛戒护琛屽伐鍏凤紝鐢ㄤ簬璋冪敤鎵樼 SciNet API 瀹屾垚鏂囩尞椹卞姩鐨勭鐮斿伐浣滄祦銆?</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.14367">馃搫 arXiv</a>
  路
  <a href="http://scinet.openkg.cn/register">馃攽 鑾峰彇 API Token</a>
  路
  <a href="http://scinet.openkg.cn/healthz">馃┖ API 鍋ュ悍妫€鏌?/a>
</p>

---

## 馃搼 鐩綍

- [鉁?椤圭洰姒傝](#-椤圭洰姒傝)
- [馃殌 蹇€熷紑濮媇(#-蹇€熷紑濮?
- [馃攽 API Token](#-api-token)
- [馃 SciNet 鑳藉仛浠€涔圿(#-scinet-鑳藉仛浠€涔?
- [馃З 鏀寔浠诲姟](#-鏀寔浠诲姟)
- [馃洜锔?CLI 浼樺厛宸ヤ綔娴乚(#锔?cli-浼樺厛宸ヤ綔娴?
- [馃О 鍙紪杈?Skills](#-鍙紪杈?skills)
- [馃悕 Python SDK](#-python-sdk)
- [鈿欙笍 閰嶇疆璇存槑](#锔?閰嶇疆璇存槑)
- [馃И 绀轰緥鍛戒护](#-绀轰緥鍛戒护)
- [馃摝 杈撳嚭涓庤繍琛屼骇鐗(#-杈撳嚭涓庤繍琛屼骇鐗?
- [馃洜锔?PDF 宸ヤ綔娴佷腑鐨?GROBID](#锔?pdf-宸ヤ綔娴佷腑鐨?grobid)
- [馃搨 浠撳簱缁撴瀯](#-浠撳簱缁撴瀯)
- [馃Н 甯歌闂](#-甯歌闂)
- [馃椇锔?Roadmap](#锔?roadmap)
- [鉁嶏笍 Citation](#锔?citation)
- [馃搫 License](#-license)

---

## 鉁?椤圭洰姒傝

SciNet 鏄竴涓ぇ瑙勬ā銆佸瀛︾銆佸紓鏋勫鏈祫婧愮煡璇嗗浘璋憋紝鏃ㄥ湪鏋勫缓鍏ㄦ櫙寮忕瀛︽紨鍖栫綉缁溿€傚畠鏁村悎璁烘枃銆佷綔鑰呫€佸叧閿瘝銆佸紩鐢ㄣ€佸绉戠瓑瀛︽湳瀹炰綋锛屼负绉戝鍙戠幇 Agent 鍜岀鐮旇緟鍔╃郴缁熸彁渚涚粨鏋勫寲鐨勭煡璇嗗浘璋辨绱㈣兘鍔涖€?
鏈粨搴撴彁渚涢潰鍚戠敤鎴风殑杞婚噺绾?**SciNet 瀹㈡埛绔寘**銆傜敤鎴峰彲浠ラ€氳繃 pip 瀹夎锛屽湪鏈湴璋冪敤鎵樼鐨?SciNet / KG2API 鍚庣锛屾棤闇€杩炴帴 Neo4j銆佸浘鏁版嵁搴撴垨浠讳綍鍚庣鍩虹璁炬柦銆?
瀹㈡埛绔礋璐ｏ細

- 鏋勯€犵粨鏋勫寲妫€绱㈣姹傦紱
- 璋冪敤鎵樼 SciNet API锛?- 杈撳嚭绠€娲佺粓绔粨鏋滐紱
- 淇濆瓨鍙鐜板疄楠屼骇鐗╋紝濡?`request.json`銆乣response.json`銆乣report.md`锛?- 灏嗕笅娓哥鐮斿伐浣滄祦灏佽涓哄彲缂栬緫 CLI **skills**銆?
---

## 馃殌 蹇€熷紑濮?
### 1. 瀹夎

浠?GitHub 鐩存帴瀹夎锛?
```bash
pip install "git+https://github.com/zjunlp/SciNet.git#subdirectory=scinet"
```

濡傛灉鍙笇鏈涢殧绂诲畨瑁?CLI锛?
```bash
pipx install "git+https://github.com/zjunlp/SciNet.git#subdirectory=scinet"
```

瀹夎鍚庢鏌ワ細

```bash
scinet -h
```

### 2. 娉ㄥ唽 API Token

璁块棶锛?
```text
http://scinet.openkg.cn/register
```

瀹屾垚閭楠岃瘉鐮佹敞鍐岋紝骞跺鍒朵釜浜?Token銆?
### 3. 閰嶇疆

Linux / macOS锛?
```bash
export SCINET_API_BASE_URL="http://scinet.openkg.cn"
export SCINET_API_KEY="your-personal-scinet-token"
```

Windows CMD锛?
```bat
set SCINET_API_BASE_URL=http://scinet.openkg.cn
set SCINET_API_KEY=your-personal-scinet-token
```

### 4. 娴嬭瘯

```bash
scinet health
scinet config
```

### 5. 杩愯璁烘枃妫€绱?
```bash
scinet search-papers \
  --query "open world agent" \
  --keyword "high:open world agent" \
  --top-k 3
```

---

## 馃攽 API Token

SciNet 瀵瑰叕寮€鐢ㄦ埛浣跨敤涓汉 API Token銆?
### 娴忚鍣ㄦ敞鍐?
璁块棶锛?
```text
http://scinet.openkg.cn/register
```

娴佺▼锛?
1. 杈撳叆濮撳悕銆侀偖绠便€佹満鏋勫拰浣跨敤鐩殑锛?2. 鐐瑰嚮 **Send code**锛?3. 鏌ユ敹閭楠岃瘉鐮侊紱
4. 杈撳叆楠岃瘉鐮佸苟鍒涘缓 Token锛?5. 澶嶅埗杩斿洖鐨?`scinet_xxx` Token銆?
Token 鍙樉绀轰竴娆★紝璇峰Ε鍠勪繚瀛樸€?
### 鏌ヨ Token 鐘舵€?
```bash
curl -H "Authorization: Bearer $SCINET_API_KEY" \
  http://scinet.openkg.cn/v1/auth/token/status
```

### 鏌ヨ鐢ㄩ噺

```bash
curl -H "Authorization: Bearer $SCINET_API_KEY" \
  "http://scinet.openkg.cn/v1/auth/usage?days=7"
```

---

## 馃 SciNet 鑳藉仛浠€涔?
SciNet 闈㈠悜绉戠爺浠诲姟锛岃€屼笉鍙槸鏅€氬叧閿瘝妫€绱€?
1. **Search + KG Retrieval**锛氬熀浜庡叧閿瘝銆佽涔夈€佹爣棰橀敋鐐广€佸弬鑰冩枃鐚拰鍥句紶鎾绱㈢浉鍏宠鏂囥€?2. **绉戠爺宸ヤ綔娴佽嚜鍔ㄥ寲**锛氭敮鎸佹枃鐚患杩般€乮dea grounding銆乮dea evaluation銆乮dea generation銆佽秼鍔垮垎鏋愩€佺浉鍏充綔鑰呮绱㈠拰鐮旂┒鑰呯敾鍍忋€?3. **Agent 鍙嬪ソ鐨勮緭鍑?*锛氭瘡娆¤繍琛岄兘浼氫繚鐣欐満鍣ㄥ彲璇?JSON 浜х墿鍜岄潰鍚戠敤鎴风殑 Markdown 鎶ュ憡銆?4. **鍙紪杈?Skills**锛氬父鐢ㄤ笅娓镐换鍔″彲浠ュ皝瑁呬负 JSON skill锛岀敤鎴峰彲鏌ョ湅銆佸鍒躲€佷慨鏀瑰苟閫氳繃 CLI 涓€閿繍琛屻€?
---

## 馃З 鏀寔浠诲姟

| 鍛戒护 | 鍦烘櫙 | 涓昏杈撳嚭 |
|---|---|---|
| `scinet search-papers` | 璁烘枃妫€绱?| 鐩稿叧璁烘枃鍜?Markdown 鎶ュ憡 |
| `scinet related-authors` | 鐩稿叧浣滆€呭彂鐜?| 鍊欓€変綔鑰呬笌鍒嗘暟 |
| `scinet author-papers` | 浣滆€呰鏂囨煡璇?| 鎸囧畾浣滆€呰鏂?|
| `scinet support-papers` | 鏀拺璁烘枃妫€绱?| 鍊欓€変綔鑰呯殑鐩稿叧璇佹嵁璁烘枃 |
| `scinet paper-search` | 杞婚噺搴曞眰璁烘枃妫€绱?| 蹇€熻鏂囧€欓€?|
| `scinet literature-review` | 鏂囩尞缁艰堪 | 鏍稿績璁烘枃姹犮€佹椂闂寸嚎銆佸啓浣滄彁绀?|
| `scinet idea-grounding` | idea 瀹氫綅 | 鐩镐技宸ヤ綔鍜屽樊寮傚寲璇佹嵁 |
| `scinet idea-evaluate` | idea 璇勪及 | 鏂伴鎬с€佸彲琛屾€с€佸彲闈犳€ц瘉鎹?|
| `scinet idea-generate` | idea 鐢熸垚 | 涓婚缁勫悎鍜?idea seeds |
| `scinet trend-report` | 瓒嬪娍鍒嗘瀽 | 鍙戝睍鑴夌粶鍜屼唬琛ㄥ伐浣?|
| `scinet researcher-review` | 鐮旂┒鑰呰儗鏅患杩?| 鐮旂┒杞ㄨ抗涓庝唬琛ㄨ鏂?|
| `scinet skill` | 鍙紪杈?skill 娉ㄥ唽琛?| 鍙鐢ㄥ伐浣滄祦棰勮 |

---

## 馃洜锔?CLI 浼樺厛宸ヤ綔娴?
```bash
scinet -h
scinet search-papers -h
scinet literature-review -h
scinet skill -h
```

鍩虹妫€绱細

```bash
scinet search-papers \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --top-k 3 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0
```

### 妫€绱㈡ā寮?
| 妯″紡 | 鍚箟 | 閫傜敤鍦烘櫙 |
|---|---|---|
| `keyword` | 鍏抽敭璇嶉┍鍔?KG 妫€绱?| 鏈鏄庣‘ |
| `semantic` | 璇箟妫€绱?| 瀹芥硾璇箟鍖归厤 |
| `title` | 鏍囬閿氱偣妫€绱?| 宸茬煡浠ｈ〃璁烘枃 |
| `hybrid` | 鍏抽敭璇?+ 璇箟 + 鏍囬 + 鍥炬父璧?| 榛樿鎺ㄨ崘 |

鏈寚瀹?`--retrieval-mode` 鏃讹紝榛樿浣跨敤 `hybrid`銆?
### 涓撳閿氱偣

```bash
--keyword "high:open world agent"
--title "middle:Voyager: An Open-Ended Embodied Agent with Large Language Models"
--reference "low:JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models"
```

### 鍥炬绱㈠亸缃?
| 鍙傛暟 | 鍚箟 |
|---|---|
| `--bias-keyword` | 鍏抽敭璇嶈矾寰勫己搴?|
| `--bias-non-seed-keyword` | 闈炵瀛愬叧閿瘝鎵╁睍 |
| `--bias-citation` | 寮曠敤杈瑰己搴?|
| `--bias-related` | 璁烘枃鐩稿叧杈瑰己搴?|
| `--bias-authorship` | 浣滆€?璁烘枃鍏崇郴寮哄害 |
| `--bias-coauthorship` | 鍚堜綔鑰呯綉缁滃己搴?|
| `--bias-cooccurrence` | 鍏抽敭璇嶅叡鐜板己搴?|
| `--bias-exploration` | 鍥炬帰绱㈢▼搴?|
| `--ranking-profile` | 鎺掑簭鍋忓ソ锛歚precision`銆乣balanced`銆乣discovery`銆乣impact` |

---

## 馃О 鍙紪杈?Skills

SciNet skills 鏄笅娓哥鐮斿伐浣滄祦鐨?JSON 棰勮锛屾柟渚跨敤鎴锋煡鐪嬨€佸鐢ㄥ拰鑷畾涔夈€?
```bash
scinet skill list
scinet skill show literature-review
scinet skill run literature-review --query "open world agent" --keyword "high:open world agent"
scinet skill run --dry-run literature-review --query "open world agent" --keyword "high:open world agent"
```

鍒涘缓鑷畾涔?skill锛?
```bash
scinet skill init my-review --from literature-review
```

瀹冧細鐢熸垚锛?
```text
./skills/my-review.json
```

鐢ㄦ埛鍙互鐩存帴淇敼 JSON锛岀劧鍚庤繍琛岋細

```bash
scinet skill run my-review --query "your topic"
```

---

## 馃悕 Python SDK

```python
from scinet import SciNetClient

client = SciNetClient()
print(client.health())

result = client.search_papers(
    query="open world agent",
    keywords=[{"text": "open world agent", "score": 10}],
    top_k=3,
)
print(result)
```

涔熷彲浠ョ洿鎺ヤ紶鍏ラ厤缃細

```python
client = SciNetClient(
    base_url="http://scinet.openkg.cn",
    api_key="your-personal-scinet-token",
)
```

---

## 鈿欙笍 閰嶇疆璇存槑

```env
SCINET_API_BASE_URL=http://scinet.openkg.cn
SCINET_API_KEY=your-personal-scinet-token
SCINET_TIMEOUT=900
SCINET_RUNS_DIR=./runs
```

鍏煎鏃у彉閲忥細

```env
KG2API_BASE_URL=http://scinet.openkg.cn
KG2API_API_KEY=your-personal-scinet-token
```

鏂扮敤鎴锋帹鑽愪娇鐢?`SCINET_*`銆?
---

## 馃И 绀轰緥鍛戒护

### 鏂囩尞缁艰堪

```bash
scinet literature-review \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 5
```

### Idea Evaluation

```bash
scinet idea-evaluate \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:idea evaluation" \
  --keyword "middle:LLM as a judge" \
  --top-k 3
```

### Trend Report

```bash
scinet trend-report \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 5
```

---

## 馃摝 杈撳嚭涓庤繍琛屼骇鐗?
缁堢榛樿杈撳嚭绠€娲佽〃鏍硷紝瀹屾暣缁撴灉淇濆瓨鍦細

```text
runs/<run_id>/
```

甯歌鏂囦欢锛?
| 鏂囦欢 | 璇存槑 |
|---|---|
| `plan.json` | 缁撴瀯鍖栨绱㈣鍒?|
| `request.json` | 鍙戦€佺粰 SciNet API 鐨勫畬鏁磋姹?|
| `response.json` | 鍚庣鍘熷鍝嶅簲 |
| `summary.txt` | 绠€鐭憳瑕?|
| `report.md` | 闈㈠悜鐢ㄦ埛鐨?Markdown 鎶ュ憡 |
| `metadata.json` | 杩愯鍏冧俊鎭?|

---

## 馃洜锔?PDF 宸ヤ綔娴佷腑鐨?GROBID

GROBID 鐢ㄤ簬浠庣瀛?PDF 涓娊鍙栨爣棰樸€佷綔鑰呫€佹憳瑕佸拰鍙傝€冩枃鐚瓑缁撴瀯鍖栦俊鎭紝鍙湪 PDF 杈撳叆宸ヤ綔娴佷腑闇€瑕併€?
```bash
docker pull lfoppiano/grobid:latest
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:latest
curl http://127.0.0.1:8070/api/isalive
```

閰嶇疆锛?
```env
GROBID_BASE_URL=http://127.0.0.1:8070
```

---

## 馃搨 浠撳簱缁撴瀯

```text
SciNet/
  pyproject.toml
  README.md
  README_zh.md
  README_skills.md
  .env.example
  src/
    scinet/
      __init__.py
      cli.py
      client.py
      config.py
      skills.py
      builtin_skills.json
  examples/
    search_papers.sh
    literature_review.sh
    idea_evaluate.sh
  tests/
    test_import.py
  references/
    search/
```

---

## 馃Н 甯歌闂

### `scinet health` 鎴愬姛锛屼絾 `search-papers` 杩斿洖 401

璇存槑 Token 缂哄け鎴栨棤鏁堛€?
```bash
echo $SCINET_API_KEY
export SCINET_API_KEY="your-personal-scinet-token"
```

Windows CMD锛?
```bat
set SCINET_API_KEY=your-personal-scinet-token
```

### 娌℃湁鏀跺埌閭楠岃瘉鐮?
璇锋鏌ラ偖绠卞湴鍧€銆佸瀮鍦鹃偖浠跺拰楠岃瘉鐮侀噸鍙戦棿闅斻€?
### 妫€绱㈠緢鎱㈡垨瓒呮椂

浣跨敤杞婚噺鍙傛暟锛?
```bash
--top-k 3
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

### Windows 涓婃壘涓嶅埌 `scinet` 鍛戒护

```bat
.venv\Scripts\scinet.exe -h
```

鎴栭噸鏂板畨瑁咃細

```bat
.venv\Scripts\python.exe -m pip install -e .
```

---

## 馃椇锔?Roadmap

- [ ] 鍙戝竷 PyPI 鍖咃紝鏀寔 `pip install scinet-client`
- [ ] 澧炲姞 `scinet auth login/status/usage`
- [ ] 澧炲姞鏇村鍐呯疆 agent skills
- [ ] 鏀寔 Token 閲嶇疆鍜屽悐閿€
- [ ] API Playground
- [ ] MCP / Agent Runtime 闆嗘垚
- [ ] 鎵╁睍璁烘枃涔嬪鐨勭煡璇嗙被鍨嬶紝濡傛暟鎹泦銆佷唬鐮併€佹爣鍑嗐€佸畾鐞嗗拰瀹為獙缁忛獙
- [ ] 寤虹珛闈㈠悜绉戝鐮旂┒浠诲姟鐨勮瘎娴嬪熀鍑?- [ ] 鏇寸郴缁熺殑鍔ㄦ€佺煡璇嗘洿鏂版満鍒?
---

## 鉁嶏笍 Citation

濡傛灉 SciNet 瀵逛綘鐨勭爺绌舵湁甯姪锛岃寮曠敤锛?
```bibtex
@article{scinet2026,
  title={SciNet: A Large-Scale Knowledge Graph for Automated Scientific Research},
  author={SciNet Team},
  journal={arXiv preprint arXiv:2602.14367},
  year={2026}
}
```

---

## 馃搫 License

鏈」鐩噰鐢?MIT License銆傝瑙?[LICENSE](LICENSE)銆?

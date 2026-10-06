# Project Brief: RAG Assistant for the Azerbaijan Labor Code

> A RAG assistant that answers questions about the Labor Code of the Republic of
> Azerbaijan in Azerbaijani, always citing the exact articles — built and measured as a
> low-resource-language retrieval problem.

**Məqsəd:** AI Engineer interview-ları üçün portfolio layihəsi.
**Vaxt büdcəsi:** ~25-30 saat. **Dil:** Python. **Repo, kod, README:** ingiliscə.
**Data və istifadəçi sualları:** Azərbaycan dilində.

---

## Niyə bu layihə

- **Domeni başa düşürəm:** golden set-i özüm yaza və cavabları yoxlaya bilərəm.
- **Real istifadəçilər:** işçilər, HR, mühasiblər (məzuniyyət, işdən çıxma, kompensasiya).
- **Real texniki problem:** Azərbaycan dili low-resource dildir; embedding modellərinin
  keyfiyyəti ölçülməlidir, fərz edilməməlidir.
- **Lüğət uçurumu:** istifadəçi "işdən qovdular" yazır, qanun "əmək müqaviləsinin
  ləğvi" deyir. Bu, query rewriting və hybrid search üçün təbii motivasiyadır.
- **Fərqləndirici:** encoder müqayisəsi + ablation cədvəli + citation accuracy.
- **Qlobal auditoriya üçün:** README ingiliscə, nümunə suallar tərcümə ilə; layihə
  "low-resource language RAG" kimi təqdim olunur.
- **Əvvəlki seçim (GitLab Handbook) dəyişdirildi:** domeni yaxşı başa düşmürdüm,
  ona görə golden set-i keyfiyyətlə yaza bilməzdim. (ADR-000)

---

## Scope

### ✅ Daxildir
1. Yalnız Əmək Məcəlləsi (bir qanun), rəsmi mənbədən; hər chunk-da maddə nömrəsi və link
2. Maddə səviyyəsində deterministik chunking (maddə → bəndlər), metadata ilə
3. 2-3 embedding modelinin müqayisəsi (məs. OpenAI, multilingual-e5, bge-m3)
4. Hybrid search (vector + BM25) + multilingual cross-encoder rerank
5. Query rewriting: danışıq dilindən hüquqi terminlərə (orijinal + yenidən yazılmış sual)
6. Hər cavabda maddə istinadı; kontekstdə yoxdursa "bilmirəm"
7. Eval: retrieval + cavab keyfiyyəti + citation accuracy + refusal + latency/cost
8. Sadə UI (Gradio), Docker, mümkünsə canlı demo
9. README: dizayn qərarları, nəticələr, "nə işləmədi", "hüquqi məsləhət deyil" qeydi

### ❌ Daxil deyil
Digər qanunlar (Vergi Məcəlləsi və s.), qanunun tarixi versiyaları/dəyişikliklər tarixçəsi,
agentic RAG, graph DB, fine-tuning, auth, mikroservislər, React.
(Versiyalar və digər qanunlar → README "Future work".)

---

## Arxitektura (ilkin)

```
INGEST:  rəsmi mətn → maddələrə parse → chunk (maddə/bənd) → metadata
         (article_no, title, chapter, url) → embedding → Chroma  +  BM25 index

QUERY:   question
         → rewrite (danışıq dili → hüquqi terminlər)
         → vector + BM25 (orijinal VƏ yenidən yazılmış sual ilə)
         → birləşdir (RRF) + təkrarları at
         → cross-encoder rerank (orijinal suala görə) → top-k
         → LLM cavab, maddə istinadları ilə → istinadların yoxlanması
```

---

## Eval planı

**Golden set** (`eval/golden/*.jsonl`), hər sətir:
`{id, question, expected_answer, relevant_chunks, category, verified, notes}` (ADR-006;
maddə səviyyəsi `relevant_chunks`-dan çıxarılır)

| Kateqoriya | Nümunə |
|---|---|
| `factual` | "Əmək məzuniyyəti minimum neçə gündür?" |
| `exact_article` | "114-cü maddə nə deyir?" |
| `colloquial` | "Məni işdən qovdular, mənə pul verməlidirlər?" |
| `multi_article` | Cavab bir neçə maddəni birləşdirməyi tələb edir |
| `follow_up` | "Bəs hamilə qadınlar üçün?" |
| `out_of_scope` | Qanunda olmayan sual → "bilmirəm" gözlənilir |

| Növ | Metriklər |
|---|---|
| Retrieval | MRR, Recall@k, Precision@k (maddə səviyyəsində) |
| Cavab | LLM judge (accuracy, completeness, relevance); judge modeli SABİT |
| Grounding | Citation accuracy (istinad edilən maddə cavabı həqiqətən ehtiva edir?), out_of_scope refusal rate |
| Əməliyyat | p50/p95 latency, sorğu başına cost |

**Qaydalar:** dev/test bölgüsü; hər dəfə bir dəyişiklik; hər run `eval/results/`-ə;
hər konfiqurasiya ən az 2 dəfə; golden set nəticəni yaxşılaşdırmaq üçün dəyişdirilmir.

---

## Mərhələlər

- [x] **0. Scaffolding:** struktur, config, logging, lint, test
- [x] **1. Data:** mənbə və istifadə şərtləri, parse, maddə chunking, testlər (~5 saat)
- [x] **2. Baseline:** yalnız vector search (1 encoder) + cavab + istinad + sadə UI (~4 saat)
- [x] **3. Golden set + eval:** 67 sual, baseline nəticəsi (~6 saat) — ADR-006, ADR-007
- [ ] **4. Eksperimentlər:** encoder müqayisəsi → hybrid → rerank → rewriting, hər biri ayrıca (~7 saat)
- [ ] **5. Grounding:** citation yoxlaması, refusal, injection testləri (~3 saat)
- [ ] **6. Təhvil:** README, ablation cədvəli, GIF demo, Docker, deploy (~4 saat)

---

## Risklər
- Mənbə saytının istifadə şərtlərini yoxla; mənbəni README-də göstər.
- Qanun mətni dəyişə bilər: download tarixini metadata-da və README-də qeyd et.
- Hüquqi cavablar: UI və README-də "bu, hüquqi məsləhət deyil" xəbərdarlığı.
- Azərbaycan hərfləri və böyük/kiçik hərf çevrilməsi BM25-i sındıra bilər (bax CLAUDE.md §6).

---

## Qərarlar jurnalı (ADR)
Format: **Qərar → Alternativlər → Niyə → Ölçülən nəticə.**

### ADR-000: Domen GitLab Handbook-dan Əmək Məcəlləsinə dəyişdirildi
- Alternativ: GitLab Handbook üzərində permission-aware RAG
- Niyə: domeni başa düşmədən golden set-i keyfiyyətlə yazmaq və cavabları yoxlamaq mümkün deyil
- Nəticə: layihə yeni repo-da sıfırdan başladı; scaffolding və mühəndislik qaydaları saxlanıldı

### ADR-001: Üstündən xətt çəkilmiş (`<s>`) mətn parse zamanı atılır
- Qərar: rəsmi HTML-də dəyişikliklərlə çıxarılmış mətn silinmir, `<s>` ilə xətli
  saxlanılır. Parser bu mətni atır; tamamilə xətli maddələr (241, 298) nəticəyə düşmür.
- Alternativlər: xətli mətni saxlamaq; çıxarılmış maddələri `repealed: bool` field ilə saxlamaq
- Niyə: xətli mətn qüvvədə olan qanun deyil. Saxlansa, assistent onu sitat gətirər
  (grounding qaydası, CLAUDE.md §4). Ayrıca field model-i mürəkkəbləşdirir; "298-ci maddə"
  sualına cavab isə Phase 5-də refusal kimi yoxlanıla bilər.
- Nəticə: qanun body-sində 140 `<s>` tag; 329 əvəzinə 327 maddə (1-317 + 12 tireli,
  241 və 298 çıxmaqla); "ləğv edilmişdir" bəndləri (5 ədəd) ayrıca `repealed_points`-də qalır.

### ADR-002: Mənbə: e-qanun.az-dakı konsolidasiya olunmuş HTML
- Qərar: `https://frameworks.e-qanun.az/46/f_46943.html` (UI: `https://e-qanun.az/framework/46943`),
  Ədliyyə Nazirliyinin rəsmi hüquqi aktlar bazası; bütün dəyişikliklər daxil edilmiş cari mətn.
  `make data` faylı xam bayt kimi saxlayır, `.meta.json`-da URL, download tarixi, sha256,
  `Last-Modified` qeyd olunur.
- Alternativlər: rəsmi qəzet (Azərbaycan qəzeti / Qanunvericilik Toplusu) PDF-ləri — dəyişikliklər
  ayrı-ayrı aktlardadır, konsolidasiya bizim işimiz olardı; üçüncü tərəf saytlarındakı surətlər —
  aktuallığı və dəqiqliyi zəmanətsiz; e-qanun.az-ın JS API-si — sənədləşdirilməyib, dəyişə bilər.
- Niyə: rəsmi, konsolidasiya olunmuş, bir statik fayl; `<s>` ilə çıxarılmış mətn işarələnib (ADR-001),
  struktur (bölmə/fəsil/maddə/bənd) deterministik parse olunur.
- Risk: istifadə şərtləri saytda göstərilməyib. Ona görə tam mətn repo-da yayılmır (`data/` git-ignored),
  yalnız kiçik test fixture-ı commit olunur; README mənbəni və download tarixini göstərir.
- Nəticə (2026-10-05 download, `Last-Modified: 17 Sep 2026`): 329 maddə başlığı (1-317 + 12 tireli),
  327-si qüvvədə (241, 298 tamamilə xətli); 5 "ləğv edilmişdir" bəndi.

### ADR-003: Chunking: maddənin hər top-level bəndi bir chunk
- Qərar: `ingest/chunk.py`. Top-level bənd: `2.`, `2-1.`, və maddə nömrəsi prefiksli `7-1.1.`
  (yalnız prefiks dəqiq `<maddə_no>.` olduqda və ardınca rəqəm gəldikdə silinir). Dərin bəndlər
  (`7-1.1.1.`, `2-3.1.`), hərfli yarımbəndlər (`a)` … `ç)`), nömrəsiz siyahı sətirləri və maddənin
  sonundakı `Qeyd:` öz bəndinin içində qalır. Nömrəli bəndi olmayan maddə — bir chunk.
  "Ləğv edilmişdir" bəndləri chunk vermir. `chunk_id` = `<maddə>.<bənd>` (`114.2`, `3.2-1`) və ya
  `<maddə>`; bənd nömrəsi mətndən götürülür, sıra ilə verilmir (xətli bəndlər sıranı pozur: 179-da
  yalnız `2.` qalıb). Hər chunk `Maddə <no>. <başlıq>` header-i ilə başlayır.
- Alternativlər: bütöv maddə bir chunk (max 7 115 simvol, bir neçə mövzu bir vektorda qarışır,
  `Maddə 114.2` dəqiqliyində istinad mümkün olmur); sabit ölçülü (token) pəncərələr — bəndləri
  kəsir, istinad sərhədləri itir; hər abzas/yarımbənd ayrıca — `a)` sətri girişsiz mənasızdır.
- Niyə: bənd hüquqi istinadın təbii vahididir (`Maddə 114.2`), citation yoxlaması və golden set
  eyni vahidlə işləyir; deterministik, LLM-siz; header bəndi tək götürüləndə də kontekst verir.
  `Qeyd:` hər chunk-a kopyalanmır (12 maddə, sadəlik); lazım olsa Phase 4-də ölçülür.
- Nəticə (tam mətn): 327 maddə → **988 chunk** (76 maddə tək chunk), ID-lər unikal, hər maddənin
  ən az bir chunk-ı var. Ölçü (header daxil, simvol): min 99, p50 326, p95 880, p99 1 860,
  max 3 339; 7 chunk > 2 000 (179.2, 9, 31.2, 12.1, 74.1, 209, 130) — hamısı uzun `a)…` siyahılarıdır.
- Açıq təklif (əlavə edilməyib): 2 000 simvoldan uzun hərfli siyahını giriş cümləsini təkrarlayaraq
  qruplara bölmək. Phase 4-də bu chunk-ların retrieval nəticəsinə görə qərar veriləcək.

### ADR-004: Model provider: yalnız OpenAI, modellər `config.yaml`-da pin olunub
- Qərar: embedding `text-embedding-3-small`, cavab `gpt-4.1-mini-2025-04-14`, judge
  `gpt-4.1-2025-04-14` (SABİT, Phase 3-dən etibarən dəyişdirilmir). Hər üçü 2026-10-06-da
  `models.list()` ilə yoxlanılıb; heç biri deprecations siyahısında deyil. Qiymətlər ($/1M token,
  input/output): 0.02 / —, 0.40 / 1.60, 2.00 / 8.00 — `config.yaml`-dakı price table-dan
  `cost_usd` hesablanır (cached-input endirimi nəzərə alınmır, yəni rəqəm yuxarı həddir).
  `llm.py` `openai` SDK-nı konfiqurasiya olunan `base_url` ilə işlədir; SDK retry-ları söndürülüb
  (`max_retries=0`), retry siyasəti yalnız `tenacity`-dədir (429, 5xx, timeout, connection;
  exponential backoff + jitter; auth/validation xətaları dərhal `LLMError`).
- Alternativlər: Ollama/lokal modellər (pulsuz, amma Azərbaycan dilində keyfiyyət və structured
  output zəmanəti zəif, Mac-da latency yüksək; scope-dan kənar); digər API-lər (Anthropic, Gemini —
  ikinci açar və ikinci SDK, Phase 2 üçün fayda yoxdur); `gpt-5-mini` (daha ucuz input, amma
  reasoning modelidir, `temperature` dəyişdirilə bilmir — judge üçün `temperature=0` tələbimizə
  (CLAUDE.md §5) uyğun deyil); `gpt-4o-mini` (daha ucuz, Azərbaycan dilində daha zəif);
  `text-embedding-3-large` və bge-m3 — Phase 4 müqayisəsi.
- Niyə: bir provider, bir açar, structured outputs, ucuz baseline. `base_url` sayəsində
  OpenAI-compatible lokal provider Phase 4-də kod dəyişmədən qoşula bilər.
- Məlum risklər: (1) **judge və cavab modeli eyni ailədəndir (GPT-4.1)** — self-preference bias
  mümkündür: judge öz ailəsinin üslubunu yüksək qiymətləndirə bilər. İndilik qəbul edirik, çünki
  layihə OpenAI-only-dir və judge üçün `temperature=0` lazımdır; Phase 3-də judge balları əl ilə
  yoxlanmış nümunə ilə kalibrə olunacaq, nəticələr README-də bu qeydlə veriləcək. (2) Eyni ailədən
  `gpt-4.1-nano` 2026-10-23-də bağlanır — ailə həyat dövrünün sonundadır; judge dəyişsə, bütün
  judge balları yenidən hesablanmalıdır (ADR ilə).
- Ölçülən nəticə: index 988 chunk → **$0.0043** (~213k token, 10.9 s); təkrar `make index` —
  0 embedding çağırışı (988 cache hit). Sorğu başına ~$0.0006–0.0011, latency 1.3–5 s
  (smoke run, Phase 2 PR). Phase 2-nin ümumi xərci ≈ **$0.02** ($2 büdcədən).

### ADR-005: Chroma persistent + cosine + embedding modeli başına bir collection; prompt və istinad formatı
- Qərar:
  - Chroma `PersistentClient` (`data/chroma`), collection adı modeli daxil edir
    (`labor_code__text-embedding-3-small`), distance **cosine** explicit verilir (Chroma-nın
    default-u L2-dir) və açılışda yoxlanılır. Embedding-ləri özümüz ötürürük
    (`embedding_function=None`). Upsert `chunk_id` ilə — idempotent; `chunks.jsonl`-dan itən
    chunk-lar silinir. Metadata-da `point=None` `""` kimi saxlanılır (Chroma `None` qəbul etmir).
  - Embedding cache: SQLite, açar `(embedding_model, sha256(text))`.
  - Prompt (`answer-v1`): Azərbaycan dilində system prompt; hər chunk
    `<article id="114.2" title="...">…</article>` içində, mətn və atributlar `html.escape` olunur —
    chunk-dakı `</article>` delimiter-i bağlaya bilmir; içindəki göstərişlərə əməl etməmək qaydası.
    Structured output `Answer {text, citations[], refused}`; `refused=true` olduqda `citations`
    boş olmalıdır (schema validator), mətn sabit refusal cümləsi ilə əvəz olunur.
  - İstinad yoxlaması generasiyadan **sonra**: mətndəki ref-lər `text.parse_article_refs` ilə
    (`114-cü maddə` daxil) + structured `citations` siyahısı; retrieve olunmuş set ilə müqayisə.
    Qayda **qəsdən yumşaqdır**: `Maddə N` — N-in hər hansı chunk-ı retrieve olunubsa etibarlıdır;
    `Maddə N.P` — `N.<P-nin top-level hissəsi>` və ya bütöv maddə chunk-ı retrieve olunubsa.
    Etibarsızlar `citation.invalid` (`level`: article/point) kimi loglanır və cavabla qaytarılır.
    Point səviyyəsində dəqiqlik Phase 3-də ölçüləcək.
- Alternativlər: L2 (OpenAI embedding-ləri normallaşdırılıb, sıralama eyni olardı, amma score
  [0,1] cosine similarity kimi şərh olunmur və başqa encoder-lərdə (Phase 4) fərq yarana bilər);
  bir collection-da bir neçə model (vektorlar müxtəlif fəzalardadır, qarışdırmaq olmaz);
  vector DB server (Qdrant/Weaviate — scope-dan kənar, 988 vektor üçün lazımsız);
  istinadları yalnız prompt ilə "tələb etmək" (yoxlama olmadan model uydurma istinad verə bilər);
  sərt point-level yoxlama (baseline-da çox false-positive; əvvəl ölçmək lazımdır).
- Niyə: lokal, serversiz, reproducible; model dəyişikliyi yeni collection + cache açarı ilə
  təhlükəsizdir; istinad yoxlaması grounding qaydasının (CLAUDE.md §4) ölçülə bilən hissəsidir.
- Ölçülən nəticə (smoke run, 5 sual, k=8, iki dəfə): **etibarsız istinad 0**; out-of-scope sual
  (gəlir vergisi) rədd edildi. Amma 4 in-scope sualdan 3-ü rədd edildi, 1-i natamam cavablandı —
  səbəb retrieval-dır, generasiya deyil: düzgün chunk top-8-də yoxdur (114.2 "21 təqvim günü" —
  "minimum" sözü 155 "minimum əmək haqqı"-nı gətirir; "114-cü maddə" — dense search nömrəni
  tutmur; "qovdular" — danışıq dili; hamilə → 79.1 yox, 240–245). Model kontekstdən kənara
  çıxmadı. Bu, Phase 4 (BM25/hybrid, query rewriting, maddə nömrəsi ilə birbaşa lookup) üçün
  əsas motivasiyadır; rəqəmlər Phase 3-də golden set ilə ölçüləcək.

### ADR-006: Golden set: 67 sual, chunk səviyyəsində gold, sahib tərəfindən yoxlanılır, sonra dondurulur
- Qərar:
  - `eval/golden/{dev,test}.jsonl`, **67 item**: factual 21, colloquial 15, exact_article 10,
    multi_article 11, out_of_scope 10 (8 HR/hüquqi, amma başqa qanun: vergi, pensiya, işsizlik
    sığortası, MMC qeydiyyatı, iş icazəsi rüsumu, kirayə, minimum əmək haqqının **məbləği**;
    2 tamamilə kənar sual). `follow_up` atlanıb: sistem single-turn-dür, kontekst ötürülmür;
    multi-turn gələcək iş kimi qalır.
  - Gold vahidi **chunk**-dır (`relevant_chunks: ["114.2"]`), çünki istinadlar bənd
    səviyyəsindədir (ADR-003); maddə səviyyəsi ondan çıxarılır, iki dəfə saxlanılmır. Gold =
    tam cavab üçün lazım olan minimal chunk dəsti; faydalı, amma vacib olmayan chunk-lar
    `notes`-dadır.
  - Split: kateqoriya üzrə stratified, **seed 20261006**, ~55/45 → dev 38 / test 29; heç bir
    eval run-dan əvvəl təyin olunub. `eval/split.py` + test faylların bu bölgüyə uyğunluğunu yoxlayır.
  - Yazılma qaydası: `chunks.jsonl`-dan chunk seçilib, sual və cavab həmin mətndən yazılıb;
    **retriever heç vaxt işə salınmayıb** (əks halda set baseline-ın artıq tapdığına doğru
    əyilərdi). 13 bölmədən 12-si əhatə olunub (XIII yoxdur), III və V bölmələrə ağırlıq var.
    Testlər: ID unikallığı, split-lər arasında sual təkrarı yoxdur, hər gold chunk `chunks.jsonl`-da var.
  - Draft LLM (Claude) tərəfindən yazılıb, hamısı `verified: false`. `verified: true`-nu yalnız
    sahib qoyur. Yoxlanılmamış set üzərində nəticələrin label-inə `_unverified` əlavə olunur,
    README cədvəli "provisional" kimi işarələnir. Yoxlamadan sonra set **dondurulur**: dəyişiklik
    üçün ADR + `make baseline` lazımdır (`golden_sha256` hər nəticədə; `make compare` fərqli
    golden-də xəbərdarlıq edir).
- Alternativlər: maddə səviyyəsində gold (sadə, amma `Maddə 114.2` dəqiqliyini ölçmür; coll-08/09
  kimi "maddə tapıldı, bənd yox" halları görünməz qalır); golden set-i retriever nəticələrindən
  qurmaq (sürətli, amma bias); 100+ sual (yoxlama yükü sahibin vaxtına sığmır); test-i ayrıca
  yazmaq (eyni müəllif, eyni üslub; seed-li split daha şəffafdır).
- Niyə: LLM-in yazdığı golden set ancaq sahib yoxladıqdan sonra "həqiqət" olur: model qanunu
  səhv oxuya, gold-u natamam seçə bilər; yoxlanılmamış rəqəm eval-ı yox, LLM-in fikrini ölçür.
- Ölçülən nəticə (baseline, provisional): dev MRR **0.525**, Recall@5 0.635, ChunkRecall@8
  0.536; ən zəif kateqoriya colloquial (dev MRR 0.226, false-refusal ~0.56). Retrieval
  deterministikdir: iki dev run arasında retrieval metriklərinin fərqi 0.000; generasiya
  metriklərində fərq var (false-refusal 0.250 / 0.312), yəni ±0.06 noise-dur.

### ADR-007: LLM judge: sabit model, temperature=0, chunk-ları görmür, insan balları ilə kalibrasiya
- Qərar:
  - Judge `gpt-4.1-2025-04-14` (ADR-004, SABİT), `temperature=0`, yalnız `llm.py` ilə, structured
    output `JudgeScore {accuracy, completeness, relevance: 1-5, rationale}`. Prompt
    `generation/prompts.py`-da, `JUDGE_PROMPT_VERSION = "judge-v1"`, hər nəticədə saxlanılır.
    Rubric ingiliscədir (evaluator təlimatıdır, istifadəçiyə göstərilmir), hər bal üçün bir
    sətirlik anchor; "yalnız reference-ə görə qiymətləndir, öz hüquqi biliyindən istifadə etmə";
    input-lar delimiter-lərdə, escape olunub, içindəki göstərişlərə əməl edilmir.
  - Judge **retrieve olunmuş chunk-ları görmür**: o, istifadəçinin oxuduğu cavabı reference ilə
    müqayisə edir. Grounding ayrıca ölçülür (citation metrikləri).
  - Skip: out_of_scope (refusal metriki ölçür); rədd edilmiş in-scope cavab — çağırış olmadan
    1/1/1. İki orta göstərilir: bütün in-scope (refusal = 1) və yalnız cavablandırılmış.
  - Kalibrasiya: `eval/calibration.jsonl` — dev-dən 15 judge olunmuş cavab, kateqoriyalar üzrə
    round-robin, seed-li; `human_accuracy` boş, yalnız sahib doldurur. `make calibrate`:
    exact agreement, ±1 agreement, Spearman (tie-lər üçün average rank; əl ilə yazılıb).
    `eval/report.py` yenidən run olunanda, id və cavab mətni dəyişməyibsə, insan balını saxlayır.
- Alternativlər: judge-a chunk-ları vermək (faithfulness ölçər, amma retrieval səhvini cavab
  keyfiyyəti ilə qarışdırar və judge "kontekstdə var" deyə səhv cavabı bəyənə bilər); başqa
  provider-dən judge (self-preference bias azalardı, amma ikinci açar/SDK, ADR-004); reference-siz
  judge (öz biliyinə əsaslanar — hüquqi sualda yoxlanıla bilməz); refused cavabları da judge-a
  göndərmək (pul və noise, nəticə məlumdur); 1-10 şkala (anchor-lar zəifləyir).
- Niyə: reference-ə əsaslanan, anchor-lu, deterministik judge reproducible-dir; eyni ailədən olduğu
  üçün (risk ADR-004) rəqəmlər ancaq insan kalibrasiyasından sonra etibarlıdır.
- Ölçülən nəticə (baseline, provisional): judge accuracy dev **3.20** (refusal = 1 daxil),
  cavablandırılmış cavablarda **4.07**; test 3.36 / 4.28. Judge çağırışı sorğu başına ~$0.002
  xərcin təxminən yarısıdır. Kalibrasiya: **pending** (insan balları boşdur).
- Açıq məsələ (Phase 4/5, bu fazada düzəldilmir): **citation format xətası.** Cavab modeli bəzən
  structured `article_no`-ya chunk id yazır (`article_no: "254.1", point: "1"` → `Maddə 254.1.1`),
  belə istinad heç vaxt maddəyə uyğun gəlmir. Bütün etibarsız istinadlar bu növdəndir
  (`invalid_not_retrieved_rate` = 0, yəni model kontekstdən kənar maddə uydurmayıb); format xətası
  olan cavablar: dev 3/22 və 4/24 cavablandırılmış, test 5/18. Metriklər bunu ayrıca göstərir
  (`invalid_format_rate`) və gold precision-da həm maddə, həm bənd səviyyəsində səhv sayır.
  Ehtimal olunan düzəliş: `Citation.article_no`-ya pattern `^\d+(-\d+)?$` + prompt-da nümunə —
  ayrıca eksperiment kimi ölçülməlidir.
- Digər müşahidə: fact-15-də gold maddə (249) retrieve olunmayıb, model 248-dən səhv nəticə
  çıxarıb ("18 yaşdan az... məhdudiyyət yoxdur"); istinad "etibarlı" sayılır, judge accuracy = 1.
  Citation yoxlaması (retrieved set) grounding-in yalnız bir hissəsini tutur.


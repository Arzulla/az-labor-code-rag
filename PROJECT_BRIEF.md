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
`{id, question, expected_answer, relevant_articles, category}`

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
- [ ] **3. Golden set + eval:** 60-80 sual, baseline nəticəsi (~6 saat)
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


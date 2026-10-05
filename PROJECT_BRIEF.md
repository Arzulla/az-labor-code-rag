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
- [ ] **1. Data:** mənbə və istifadə şərtləri, parse, maddə chunking, testlər (~5 saat)
- [ ] **2. Baseline:** yalnız vector search (1 encoder) + cavab + istinad + sadə UI (~4 saat)
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

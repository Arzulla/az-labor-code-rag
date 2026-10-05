# Test fixtures

## `labor_code_sample.html`

A small excerpt of the official Labor Code HTML, for parser tests. Byte-exact slices of
the source (windows-1251, Word-exported markup, HTML entities); nothing was decoded or
re-encoded. Slices are joined with the source's own `\r\n\r\n` paragraph separator.

- Source: https://frameworks.e-qanun.az/46/f_46943.html (UI: https://e-qanun.az/framework/46943)
- Downloaded: 2026-10-05, `Last-Modified: Thu, 17 Sep 2026 05:47:29 GMT`
- Source sha256: `4dfc57194a5ba96eada738bd6033f61f451893f269df9fe4ac9033e0136008ff`

Contents, in order:

| Slice | Why it is here |
|---|---|
| `<html>` + `<head>` up to the first `[if gte mso 9]` XML block | charset meta, `<title>`, conditional comment that must not become text |
| `<style>` cut after the first `@font-face`, then closed | CSS that must not become text (the original block is ~55 KB) |
| `<body>` + `<div class=WordSection1>` | original body markup |
| Maddə 7-1, complete | dotted sub-points (`7-1.1.1.`), endnote reference `[25]` in the heading |
| Maddə 114, points 1-3 (points 4-6 cut) | lettered sub-points `a)` … `ç)`, `d)`, `e)` |
| Maddə 124, points 1-3 (points 4-8 cut) | repealed points (`ləğv edilmişdir`), endnote reference `[351]` |
| Maddə 317 (last article) + `ƏLAVƏLƏR` heading | end of the law body; appendices start |
| `mso-element:endnote-list` div + endnote `edn140` (truncated) | a false `Maddə 49.` heading: the old wording quoted inside a footnote |

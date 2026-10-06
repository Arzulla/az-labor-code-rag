# Test fixtures

## `labor_code_sample.html`

A small excerpt of the official Labor Code HTML, for parser tests. Byte-exact slices of
the source (windows-1251, Word-exported markup, HTML entities); nothing was decoded or
re-encoded. Slices are joined with the source's own `\r\n\r\n` paragraph separator.
Heading slices include the empty Word paragraphs between a heading and its title.

- Source: https://frameworks.e-qanun.az/46/f_46943.html (UI: https://e-qanun.az/framework/46943)
- Downloaded: 2026-10-05, `Last-Modified: Thu, 17 Sep 2026 05:47:29 GMT`
- Source sha256: `4dfc57194a5ba96eada738bd6033f61f451893f269df9fe4ac9033e0136008ff`

Contents, in order:

| Slice | Why it is here |
|---|---|
| `<html>` + `<head>` up to the first `[if gte mso 9]` XML block | charset meta, `<title>`, conditional comment that must not become text |
| `<style>` cut after the first `@font-face`, then closed | CSS that must not become text (the original block is ~55 KB) |
| `<body>` + `<div class=WordSection1>` | original body markup |
| `I bölmə` + `Birinci fəsil` headings with their titles | part/chapter context for 7-1 |
| Maddə 7-1, complete | dotted sub-points (`7-1.1.1.`), endnote reference `[25]` in the heading |
| `V bölmə` + `On yeddinci fəsil` headings with their titles | part and chapter change before 114 |
| Maddə 114, points 1-3 (points 4-6 cut) | lettered sub-points `a)` … `ç)`, `d)`, `e)` |
| `On səkkizinci fəsil` heading with its title | chapter changes, part stays the same |
| Maddə 124, points 1-3 (points 4-8 cut) | repealed points (`ləğv edilmişdir`), endnote reference `[351]` |
| Maddə 298, complete | repealed article: an `<h3>` heading, everything struck through (`<s>`) except the endnote reference `[584]` |
| `XII bölmə` + `Qırx altıncı fəsil` headings with their titles | struck word (`<s>sosial</s>`) and an endnote reference inside heading titles |
| Maddə 305, complete | unnumbered body text with one struck word |
| `XIII bölmə` + `Qırx səkkizinci fəsil` headings with their titles | last part and chapter |
| Maddə 317 (last article) + `ƏLAVƏLƏR` heading | end of the law body; appendices start |
| `mso-element:endnote-list` div + endnote `edn140` (truncated) | a false `Maddə 49.` heading: the old wording quoted inside a footnote |

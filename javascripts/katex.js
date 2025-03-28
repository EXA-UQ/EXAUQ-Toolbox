// To integrate KaTeX for rendering maths. Taken from
// https://squidfunk.github.io/mkdocs-material/reference/math/#katex
// accessed 2024-08-23

document$.subscribe(({ body }) => { 
    renderMathInElement(body, {
      delimiters: [
        { left: "$$",  right: "$$",  display: true },
        { left: "$",   right: "$",   display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
    })
  })
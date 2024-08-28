md -> html:
`pandoc --standalone --include-in-header=_includes/analytics.html --mathjax -f markdown -t html 'SOTA in Data Attribution.md' -o a-new-sota-in-data-attribution.html --css=styles.css`

local server:
`bundle exec jekyll serve
`
ruby updates
`gem install`

.md files are implicitly converted to .html by jekyll. but mathjax rendering is broken there, so we use `pandoc`. 
this is why the .md input and .html output are *slightly* differently named. important detail
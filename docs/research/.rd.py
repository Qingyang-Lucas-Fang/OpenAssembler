import sys,re,html
t=sys.stdin.read()
m=re.search(r'<article[^>]*markdown-body[^>]*>(.*?)</article>',t,re.S)
t=m.group(1) if m else t
t=re.sub(r'<script.*?</script>|<style.*?</style>|<svg.*?</svg>','',t,flags=re.S)
t=html.unescape(re.sub(r'<[^>]+>',' ',t)); t=re.sub(r'\s+',' ',t)
print(t[:int(sys.argv[1])])

import sys,re,html
t=sys.stdin.read()
t=re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S)
t=html.unescape(re.sub(r'<[^>]+>',' ',t)); t=re.sub(r'\s+',' ',t)
kws=sys.argv[2:]
n=int(sys.argv[1])
for k in kws:
  for m in list(re.finditer(re.escape(k),t))[:2]:
    print('##',k,':',t[max(0,m.start()-300):m.start()+n]); print()

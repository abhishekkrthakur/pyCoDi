import pygame as P
import sys
D=P.display
I=P.image
D.init()
f,g=sys.argv[1:3]
m=I.load(f)
s=D.set_mode(m.get_size())
r=k=0
c=1
while c:
 for e in P.event.get():
  t=e.type
  if t==5:x,y=e.pos;k=1
  if t==4 and k:i,j=e.pos;r=(x,y,i-x,j-y)
  if t==6:c=0
 s.blit(m,(0,0))
 if r and c:P.draw.rect(s,0,r,1)
 D.flip()
q=s.subsurface(r)
I.save(q,g)
import sys, types, numpy as np, re, ast
src=open('/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/reset/programs/reset.py').read()
a=src.index('def _plane'); b=src.index('def rz(deg)')
HALF=(0.040,0.058); EDGE_MARGIN=0.10; ns=dict(np=np,HALF=HALF,EDGE_MARGIN=EDGE_MARGIN); exec(src[a:b],ns)
y=np.radians(21.2); A=np.array([np.cos(y),np.sin(y)])
for t in ((0.28,-0.03),(0.29,-0.02),(0.30,-0.01),(0.28,-0.01),(0.30,-0.03)):
    best=min((np.linalg.norm(ns['safe_aim'](t,np.array([np.cos(y+np.radians(g)),np.sin(y+np.radians(g))]))-t),g) for g in range(-30,31,5))
    print(t,'least shift cm',round(best[0]*100,2),'at turn',best[1])

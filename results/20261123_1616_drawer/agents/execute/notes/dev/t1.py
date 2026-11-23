import json, sys
import cv2, numpy as np
W='/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute'
sys.path.insert(0, W+'/programs')
import vision as V
A=W+'/attempts/attempt_4/'
vw=json.load(open(A+'views.json'))
tz=-0.0153; face=tz+0.0915
c0=np.array([0.2855,-0.0055])
for n in ('drawer_high','drawer_low'):
    im=cv2.imread(A+n+'.jpg'); T=np.array(vw[n]['T']); o=vw[n]['crop_xy']
    d=V.find_drawer(im,T,c0,face,origin=o); print(n,d)
    if d.get('c'):
        tb=V.find_tab(im,T,d,face,origin=o); print('  tab',tb)
        R,_=V.rectify(im,T,d['c'],d['yaw']-90,(d['size'][0]/2+.01,d['size'][1]/2+.01),face,origin=o)
        if tb.get('s') is not None:
            cv2.circle(R,(int((tb['s'])/V.RES+R.shape[1]/2),int(tb['t']/V.RES+R.shape[0]/2)),6,(0,255,0),1)
        cv2.imwrite(f'/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute/dev_out/{n}_rect.jpg',R)
a,b=V.axes(21.2); nom=np.array([0.4469,-0.011]); prior=nom+0.008*a+0.011*b
for n in ('opening_high','opening_low'):
    im=cv2.imread(A+n+'.jpg'); T=np.array(vw[n]['T']); o=vw[n]['crop_xy']
    r=V.find_opening(im,T,prior,21.2,[0.0757,0.115],tz+0.095,origin=o); print(n,r)
    if r.get('c'):
        R,_=V.rectify(im,T,r['c'],r['yaw'],(0.07,0.09),tz+0.095,origin=o)
        cx,cy=(R.shape[1]-1)/2,(R.shape[0]-1)/2; hw,hh=0.0757/2/V.RES,0.115/2/V.RES
        cv2.rectangle(R,(int(cx-hw),int(cy-hh)),(int(cx+hw),int(cy+hh)),(0,255,0),1)
        cv2.imwrite(W+f'/dev_out/{n}_fit.jpg',R)

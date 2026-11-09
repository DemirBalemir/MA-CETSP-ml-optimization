from pathlib import Path
import ctypes as C,os,re,json,hashlib,time
BASE=Path(__file__).resolve().parent;O=BASE/'rerun';O.mkdir(exist_ok=True)
src=BASE/'input_model.lp';text=src.read_text();n=37
# Read geometry from exported equalities and disk constraints, asserting exact layout.
centers=[];radii=[]
for i in range(n):
 cx=float(re.search(rf'R{74+3*i}: C{111+i} \+ C{185+i} = ([^\n]+)',text)[1]);cy=float(re.search(rf'R{75+3*i}: C{148+i} \+ C{222+i} = ([^\n]+)',text)[1])
 rhs=float(re.search(rf'qc{2*i+1}: .* >= ([^\n]+)',text)[1]);assert rhs<=0
 centers.append((cx,cy));radii.append((-rhs)**.5)
zero=[i for i,r in enumerate(radii) if r==0];assert zero==[0]
# Equivalent variants, each retaining the saved geometry and tour order.
def variant(fix=False,bounds=False,distlb=False):
 s=text
 if fix:
  for i in zero:
   s=re.sub(rf'^ qc{2*i+1}:.*\n','',s,flags=re.M)
   s=s.replace('Bounds\n',f' fix_x_{i}: C{185+i} = {centers[i][0]:.17g}\n fix_y_{i}: C{222+i} = {centers[i][1]:.17g}\nBounds\n')
 if bounds:
  for i,((cx,cy),r) in enumerate(zip(centers,radii)):
   for j,c in [(185+i,cx),(222+i,cy)]:s=s.replace(f' C{j} free\n',f' {c-r:.17g} <= C{j} <= {c+r:.17g}\n')
 if distlb:
  for i in range(n):s=s.replace(f' C{i} free\n',f' C{i} >= 0\n')
 return s
cases=[('baseline',text,{}),('fix_zero_radius',variant(fix=True),{}),('coordinate_bounds',variant(bounds=True),{}),('distance_lower_bounds',variant(distlb=True),{}),('fix_zero_and_coordinate_bounds',variant(fix=True,bounds=True),{}),('baseline_numericfocus3',text,{'NumericFocus':3}),('baseline_homogeneous',text,{'BarHomogeneous':1}),('baseline_no_presolve',text,{'Presolve':0}),('coordinate_bounds_tight',variant(bounds=True),{'FeasibilityTol':1e-9,'BarQCPConvTol':1e-10})]
import sys
if len(sys.argv)>1:cases=[c for c in cases if c[0] in sys.argv[1:]]
# Use the caller's Gurobi license configuration; no license is bundled.
GH=Path(os.environ.get('GUROBI_HOME',r'C:\gurobi1203\win64'));os.add_dll_directory(str(GH/'bin'));g=C.CDLL(str(GH/'bin/gurobi120.dll'));P=C.c_void_p;I=C.c_int;D=C.c_double;S=C.c_char_p
spec={'GRBemptyenvinternal':([C.POINTER(P),I,I,I],I),'GRBstartenv':([P],I),'GRBsetintparam':([P,S,I],I),'GRBsetdblparam':([P,S,D],I),'GRBsetstrparam':([P,S,S],I),'GRBreadmodel':([P,S,C.POINTER(P)],I),'GRBoptimize':([P],I),'GRBgetintattr':([P,S,C.POINTER(I)],I),'GRBgetdblattr':([P,S,C.POINTER(D)],I),'GRBgetdblattrarray':([P,S,I,I,C.POINTER(D)],I),'GRBgeterrormsg':([P],S),'GRBwrite':([P,S],I),'GRBfreemodel':([P],None),'GRBfreeenv':([P],None)}
for k,(a,r) in spec.items():getattr(g,k).argtypes=a;getattr(g,k).restype=r
g.GRBgetvarbyname.argtypes=[P,S,C.POINTER(I)];g.GRBgetvarbyname.restype=I
results=[]
for name,lp,params in cases:
 p=O/(name+'.lp');p.write_text(lp);env=P();model=P()
 def check(code):
  if code:raise RuntimeError(str(code)+': '+str(g.GRBgeterrormsg(env)))
 try:
  check(g.GRBemptyenvinternal(C.byref(env),12,0,3));check(g.GRBsetintparam(env,b'LogToConsole',0));check(g.GRBsetstrparam(env,b'LogFile',str(O/(name+'.log')).encode()));check(g.GRBsetdblparam(env,b'TimeLimit',60.));
  for k,v in params.items():check((g.GRBsetintparam if isinstance(v,int) else g.GRBsetdblparam)(env,k.encode(),v))
  check(g.GRBstartenv(env));check(g.GRBreadmodel(env,str(p).encode(),C.byref(model)));check(g.GRBoptimize(model))
  def ai(k):v=I();check(g.GRBgetintattr(model,k.encode(),C.byref(v)));return v.value
  def ad(k):v=D();check(g.GRBgetdblattr(model,k.encode(),C.byref(v)));return v.value
  row=dict(case=name,status=ai('Status'),solutions=ai('SolCount'),runtime=ad('Runtime'),variables=ai('NumVars'),linear_constraints=ai('NumConstrs'),quadratic_constraints=ai('NumQConstrs'),params=params,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
  if row['solutions']:
   row['objective']=ad('ObjVal')
   def value(name):
    index=I();check(g.GRBgetvarbyname(model,name.encode(),C.byref(index)));assert index.value>=0
    v=(D*1)();check(g.GRBgetdblattrarray(model,b'X',index.value,1,v));return v[0]
   xy=[(value(f'C{185+i}'),value(f'C{222+i}')) for i in range(n)]
   cost=sum(((xy[i][0]-xy[(i+1)%n][0])**2+(xy[i][1]-xy[(i+1)%n][1])**2)**.5 for i in range(n));row['recomputed_cost']=cost;row['max_disk_distance_excess']=max((((x-cx)**2+(y-cy)**2)**.5-r) for (x,y),(cx,cy),r in zip(xy,centers,radii));row['depot_distance']=((xy[0][0]-100)**2+(xy[0][1]-100)**2)**.5
   for attr in ['ConstrVio','BoundVio','DualVio','ComplVio']:
    try:row[attr]=ad(attr)
    except RuntimeError:pass
   check(g.GRBwrite(model,str(O/(name+'.sol')).encode()))
  results.append(row);print(json.dumps(row),flush=True)
 finally:
  if model:g.GRBfreemodel(model)
  if env:g.GRBfreeenv(env)
 result_path=O/('results_extra.json' if len(sys.argv)>1 else 'results.json')
 result_path.write_text(json.dumps(results,indent=2))

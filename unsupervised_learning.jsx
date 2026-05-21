import { useState, useEffect, useRef, useCallback } from "react";

// ─── Utility Math ──────────────────────────────────────────────────────────────
const rand = (min, max) => Math.random() * (max - min) + min;
const dist = (a, b) => Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2);

function generateDataset(type, n = 200) {
  const pts = [];
  if (type === "blobs") {
    const centers = [[150,150],[350,120],[250,300],[450,300],[100,350]];
    for (let i = 0; i < n; i++) {
      const c = centers[i % centers.length];
      pts.push([c[0]+rand(-50,50), c[1]+rand(-50,50)]);
    }
  } else if (type === "moons") {
    for (let i = 0; i < n/2; i++) {
      const t = (i/(n/2))*Math.PI;
      pts.push([200+160*Math.cos(t)+rand(-15,15), 200+80*Math.sin(t)+rand(-15,15)]);
      pts.push([200+160*Math.cos(t+Math.PI)+rand(-15,15), 260-80*Math.sin(t)+rand(-15,15)]);
    }
  } else if (type === "circles") {
    for (let i = 0; i < n; i++) {
      const t = rand(0, 2*Math.PI);
      const r = i < n/2 ? 60 : 130;
      pts.push([250+r*Math.cos(t)+rand(-10,10), 220+r*Math.sin(t)+rand(-10,10)]);
    }
  } else {
    for (let i = 0; i < n; i++) pts.push([rand(40,500), rand(40,400)]);
  }
  return pts;
}

// ─── K-Means ──────────────────────────────────────────────────────────────────
function kMeans(pts, k, iters=50) {
  let centroids = pts.slice().sort(()=>Math.random()-0.5).slice(0,k).map(p=>[...p]);
  let labels = new Array(pts.length).fill(0);
  for (let it=0; it<iters; it++) {
    labels = pts.map(p => {
      let best=0, bd=Infinity;
      centroids.forEach((c,i)=>{ const d=dist(p,c); if(d<bd){bd=d;best=i;} });
      return best;
    });
    centroids = centroids.map((_,i) => {
      const group = pts.filter((_,j)=>labels[j]===i);
      if(!group.length) return centroids[i];
      return [group.reduce((s,p)=>s+p[0],0)/group.length, group.reduce((s,p)=>s+p[1],0)/group.length];
    });
  }
  return { labels, centroids };
}

// ─── DBSCAN ───────────────────────────────────────────────────────────────────
function dbscan(pts, eps=45, minPts=4) {
  const labels = new Array(pts.length).fill(-1);
  let cluster = 0;
  const visited = new Set();
  const neighbors = i => pts.map((_,j)=>j).filter(j=>dist(pts[i],pts[j])<=eps);
  for (let i=0; i<pts.length; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const nb = neighbors(i);
    if (nb.length < minPts) { labels[i] = -2; continue; }
    labels[i] = cluster;
    const q = [...nb];
    while (q.length) {
      const j = q.shift();
      if (!visited.has(j)) {
        visited.add(j);
        const nb2 = neighbors(j);
        if (nb2.length >= minPts) q.push(...nb2.filter(x=>!visited.has(x)));
      }
      if (labels[j] === -1 || labels[j] === -2) labels[j] = cluster;
    }
    cluster++;
  }
  return { labels, nClusters: cluster };
}

// ─── PCA (2D projection only) ─────────────────────────────────────────────────
function pca(pts) {
  const mx = pts.reduce((s,p)=>s+p[0],0)/pts.length;
  const my = pts.reduce((s,p)=>s+p[1],0)/pts.length;
  const centered = pts.map(p=>[p[0]-mx, p[1]-my]);
  const cxx = centered.reduce((s,p)=>s+p[0]*p[0],0)/pts.length;
  const cyy = centered.reduce((s,p)=>s+p[1]*p[1],0)/pts.length;
  const cxy = centered.reduce((s,p)=>s+p[0]*p[1],0)/pts.length;
  const trace = cxx+cyy, det = cxx*cyy-cxy*cxy;
  const l1 = trace/2+Math.sqrt(Math.max(0,(trace/2)**2-det));
  const l2 = trace/2-Math.sqrt(Math.max(0,(trace/2)**2-det));
  const v1 = cxy ? [l1-cyy, cxy] : [1,0];
  const n1 = Math.sqrt(v1[0]**2+v1[1]**2)||1;
  const ev1 = [v1[0]/n1, v1[1]/n1];
  const ev2 = [-ev1[1], ev1[0]];
  const proj = centered.map(p=>[ p[0]*ev1[0]+p[1]*ev1[1], p[0]*ev2[0]+p[1]*ev2[1] ]);
  const px = proj.map(p=>p[0]), py = proj.map(p=>p[1]);
  const minx=Math.min(...px),maxx=Math.max(...px),miny=Math.min(...py),maxy=Math.max(...py);
  const sx=(maxx-minx)||1, sy=(maxy-miny)||1;
  const scaled = proj.map(p=>[ 40+(p[0]-minx)/sx*460, 30+(p[1]-miny)/sy*380 ]);
  return { projected: scaled, variance: [l1/(l1+l2||1)*100, l2/(l1+l2||1)*100], labels: scaled.map((_,i)=>i%5) };
}

// ─── Hierarchical (simple single-linkage) ─────────────────────────────────────
function hierarchical(pts, k=4) {
  const n = pts.length;
  let clusters = pts.map((_,i)=>[i]);
  while (clusters.length > k) {
    let best = [0,1,Infinity];
    for (let i=0;i<clusters.length;i++)
      for (let j=i+1;j<clusters.length;j++) {
        let minD=Infinity;
        for (const a of clusters[i]) for (const b of clusters[j]) { const d=dist(pts[a],pts[b]); if(d<minD) minD=d; }
        if(minD<best[2]) best=[i,j,minD];
      }
    const merged = [...clusters[best[0]],...clusters[best[1]]];
    clusters = clusters.filter((_,i)=>i!==best[0]&&i!==best[1]);
    clusters.push(merged);
  }
  const labels = new Array(n).fill(0);
  clusters.forEach((cl,ci)=>cl.forEach(i=>{ labels[i]=ci; }));
  return { labels };
}

// ─── Color Palette ────────────────────────────────────────────────────────────
const PALETTE = ["#00d4aa","#ff6b6b","#ffd93d","#a78bfa","#38bdf8","#fb923c","#4ade80","#f472b6"];
const NOISE_COLOR = "#444";

export default function App() {
  const [algo, setAlgo] = useState("kmeans");
  const [dataset, setDataset] = useState("blobs");
  const [k, setK] = useState(4);
  const [eps, setEps] = useState(45);
  const [minPts, setMinPts] = useState(4);
  const [points, setPoints] = useState([]);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState(null);
  const [tab, setTab] = useState("viz");

  const generate = useCallback(() => {
    const pts = generateDataset(dataset);
    setPoints(pts);
    setResult(null);
    setStats(null);
  }, [dataset]);

  useEffect(() => { generate(); }, [generate]);

  const run = () => {
    setRunning(true);
    setTimeout(() => {
      let res;
      if (algo === "kmeans") res = kMeans(points, k);
      else if (algo === "dbscan") res = dbscan(points, eps, minPts);
      else if (algo === "pca") res = pca(points);
      else res = hierarchical(points, k);
      setResult(res);

      const lbls = res.labels;
      const unique = [...new Set(lbls.filter(l=>l>=0))];
      const sizes = unique.map(l=>lbls.filter(x=>x===l).length);
      const noise = lbls.filter(l=>l===-2).length;
      setStats({ clusters: unique.length, sizes, noise, points: points.length });
      setRunning(false);
    }, 200);
  };

  const vizPoints = algo==="pca" && result ? result.projected : points;
  const labels = result ? result.labels : null;

  const codeSnippets = {
    kmeans: `import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + center
    for center in [(0,0),(4,0),(2,4),(0,4),(4,4)]
])

# K-Means Clustering
kmeans = KMeans(n_clusters=5, init='k-means++',
                n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
scatter = axes[0].scatter(X[:,0], X[:,1], c=labels, cmap='tab10', alpha=0.7)
axes[0].scatter(kmeans.cluster_centers_[:,0],
                kmeans.cluster_centers_[:,1],
                s=200, c='red', marker='*', zorder=5)
axes[0].set_title(f'K-Means (k={kmeans.n_clusters})')

# Elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

axes[1].plot(range(1,11), inertias, 'bo-')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Inertia')
axes[1].set_title('Elbow Method')
plt.tight_layout()
plt.show()

print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia: {kmeans.inertia_:.2f}")`,

    dbscan: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Moons dataset (non-convex)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = labels == label
    marker = 'x' if label == -1 else 'o'
    color = 'gray' if label == -1 else color
    axes[0].scatter(X[mask, 0], X[mask, 1],
                   c=[color], marker=marker, alpha=0.7)
axes[0].set_title(f'DBSCAN: {n_clusters} clusters, {n_noise} noise')

# EPS sensitivity
cluster_counts = []
eps_range = np.arange(0.1, 1.0, 0.05)
for e in eps_range:
    db = DBSCAN(eps=e, min_samples=5)
    lbl = db.fit_predict(X)
    cluster_counts.append(len(set(lbl)) - (1 if -1 in lbl else 0))

axes[1].plot(eps_range, cluster_counts, 'go-')
axes[1].set_xlabel('EPS')
axes[1].set_ylabel('Number of Clusters')
axes[1].set_title('EPS Sensitivity')
plt.tight_layout()
plt.show()

print(f"Clusters: {n_clusters}, Noise: {n_noise}")`,

    pca: `import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset (high-dimensional)
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Full PCA for explained variance
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
scatter = axes[0].scatter(X_pca[:,0], X_pca[:,1],
                          c=digits.target, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ax=axes[0])
axes[0].set_title('PCA 2D Projection')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

# Explained variance
axes[1].bar(range(1,21), pca_full.explained_variance_ratio_[:20])
axes[1].set_title('Variance per Component')

# Cumulative
axes[2].plot(cumvar[:30], 'b-o', markersize=4)
axes[2].axhline(0.95, color='r', linestyle='--', label='95%')
axes[2].set_title('Cumulative Variance')
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"PC1+PC2 explains: {sum(pca.explained_variance_ratio_)*100:.1f}%")
n95 = np.argmax(cumvar >= 0.95) + 1
print(f"Components for 95% variance: {n95}")`,

    hierarchical: `import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=150, centers=4, cluster_std=0.8, random_state=42)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = agg.fit_predict(X)

# Dendrogram (small subset for clarity)
Z = linkage(X[:50], method='ward')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
scatter = axes[0].scatter(X[:,0], X[:,1], c=labels, cmap='tab10', alpha=0.7)
axes[0].set_title('Hierarchical Clustering (Ward)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
plt.colorbar(scatter, ax=axes[0])

# Dendrogram
dendrogram(Z, ax=axes[1], truncate_mode='level', p=5,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7*max(Z[:,2]))
axes[1].set_title('Dendrogram (Ward Linkage)')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Distance')

plt.tight_layout()
plt.show()

# Compare linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, method in zip(axes, linkage_methods):
    cl = AgglomerativeClustering(n_clusters=4, linkage=method)
    lbl = cl.fit_predict(X)
    ax.scatter(X[:,0], X[:,1], c=lbl, cmap='tab10', alpha=0.7, s=15)
    ax.set_title(f'{method.capitalize()} linkage')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Linkage Method Comparison')
plt.tight_layout()
plt.show()`
  };

  const algoInfo = {
    kmeans: { name: "K-Means", badge: "Partitional", color: "#00d4aa", desc: "Partitions data into k clusters by minimizing intra-cluster variance. Uses k-means++ initialization for optimal centroid placement.", complexity: "O(nkti)", params: ["k: Number of clusters","Iterations: convergence steps","Init: k-means++ or random"] },
    dbscan: { name: "DBSCAN", badge: "Density", color: "#ff6b6b", desc: "Density-Based Spatial Clustering. Groups dense regions, marks outliers as noise. Handles non-convex shapes perfectly.", complexity: "O(n log n)", params: ["ε (eps): neighborhood radius","minPts: core point threshold","Noise: isolated points labeled -1"] },
    pca: { name: "PCA", badge: "Dimensionality", color: "#a78bfa", desc: "Principal Component Analysis reduces high-dimensional data to 2D by projecting onto axes of maximum variance.", complexity: "O(n²d)", params: ["Components: axes of variance","Explained variance %","Eigenvectors: principal axes"] },
    hierarchical: { name: "Hierarchical", badge: "Agglomerative", color: "#ffd93d", desc: "Builds a dendrogram by iteratively merging closest clusters. No need to specify k beforehand.", complexity: "O(n² log n)", params: ["k: final cut level","Linkage: single/complete/ward","Dendrogram: tree structure"] }
  };

  const info = algoInfo[algo];

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", background: "#0d1117", minHeight: "100vh", color: "#e6edf3" }}>
      {/* Header */}
      <div style={{ background: "#161b22", borderBottom: "1px solid #30363d", padding: "12px 24px", display: "flex", alignItems: "center", gap: "16px" }}>
        <div style={{ display: "flex", gap: "8px" }}>
          <div style={{ width:12, height:12, borderRadius:"50%", background:"#ff5f57" }}/>
          <div style={{ width:12, height:12, borderRadius:"50%", background:"#ffbd2e" }}/>
          <div style={{ width:12, height:12, borderRadius:"50%", background:"#28c840" }}/>
        </div>
        <span style={{ color:"#8b949e", fontSize:13 }}>~/unsupervised-learning</span>
        <div style={{ marginLeft:"auto", display:"flex", gap:8 }}>
          {["⭐ Star","🍴 Fork"].map(b=>(
            <button key={b} style={{ background:"#21262d", border:"1px solid #30363d", color:"#e6edf3", padding:"4px 12px", borderRadius:6, fontSize:12, cursor:"pointer" }}>{b}</button>
          ))}
        </div>
      </div>

      {/* Repo title */}
      <div style={{ padding:"20px 24px 0", borderBottom:"1px solid #30363d" }}>
        <div style={{ fontSize:20, fontWeight:700, color:"#58a6ff", marginBottom:6 }}>
          🤖 unsupervised-learning-algorithms
        </div>
        <div style={{ color:"#8b949e", fontSize:13, marginBottom:16 }}>
          Interactive playground for K-Means · DBSCAN · PCA · Hierarchical Clustering
        </div>
        <div style={{ display:"flex", gap:8, marginBottom:16, flexWrap:"wrap" }}>
          {["Python","scikit-learn","NumPy","matplotlib","unsupervised-learning","clustering","pca"].map(tag=>(
            <span key={tag} style={{ background:"#132f4c", color:"#58a6ff", fontSize:11, padding:"2px 10px", borderRadius:20, border:"1px solid #1d4a8a" }}>{tag}</span>
          ))}
        </div>
        {/* Tabs */}
        <div style={{ display:"flex", gap:0, borderBottom:"none" }}>
          {[["viz","📊 Visualizer"],["code","💻 Code"],["about","📖 Theory"]].map(([id,label])=>(
            <button key={id} onClick={()=>setTab(id)} style={{ background:"none", border:"none", color: tab===id ? "#e6edf3" : "#8b949e", padding:"8px 16px", cursor:"pointer", fontSize:13, borderBottom: tab===id ? "2px solid #f78166" : "2px solid transparent", transition:"all 0.2s" }}>{label}</button>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div style={{ padding:24 }}>
        {tab === "viz" && (
          <div style={{ display:"grid", gridTemplateColumns:"280px 1fr", gap:20 }}>
            {/* Controls */}
            <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
              {/* Algorithm select */}
              <div style={{ background:"#161b22", border:"1px solid #30363d", borderRadius:8, padding:16 }}>
                <div style={{ fontSize:11, color:"#8b949e", marginBottom:10, textTransform:"uppercase", letterSpacing:1 }}>Algorithm</div>
                {Object.entries(algoInfo).map(([id,{name,badge,color}])=>(
                  <button key={id} onClick={()=>{setAlgo(id);setResult(null);}} style={{ width:"100%", display:"flex", alignItems:"center", gap:10, background: algo===id ? "#1f2937" : "transparent", border: algo===id ? `1px solid ${color}` : "1px solid transparent", color: algo===id ? color : "#8b949e", padding:"8px 12px", borderRadius:6, cursor:"pointer", marginBottom:4, fontSize:13, textAlign:"left", transition:"all 0.2s" }}>
                    <span style={{ background: algo===id ? color+"33" : "transparent", color, padding:"1px 8px", borderRadius:4, fontSize:10, fontWeight:700 }}>{badge}</span>
                    {name}
                  </button>
                ))}
              </div>

              {/* Dataset */}
              <div style={{ background:"#161b22", border:"1px solid #30363d", borderRadius:8, padding:16 }}>
                <div style={{ fontSize:11, color:"#8b949e", marginBottom:10, textTransform:"uppercase", letterSpacing:1 }}>Dataset</div>
                {[["blobs","🔵 Gaussian Blobs"],["moons","🌙 Two Moons"],["circles","⭕ Concentric"],["random","🔀 Random"]].map(([id,label])=>(
                  <button key={id} onClick={()=>setDataset(id)} style={{ width:"100%", background: dataset===id?"#1f2937":"transparent", border: dataset===id?"1px solid #30363d":"1px solid transparent", color: dataset===id?"#e6edf3":"#8b949e", padding:"6px 12px", borderRadius:6, cursor:"pointer", marginBottom:3, fontSize:12, textAlign:"left" }}>{label}</button>
                ))}
                <button onClick={generate} style={{ width:"100%", marginTop:8, background:"#21262d", border:"1px solid #30363d", color:"#e6edf3", padding:"6px 12px", borderRadius:6, cursor:"pointer", fontSize:12 }}>⟳ Regenerate</button>
              </div>

              {/* Params */}
              <div style={{ background:"#161b22", border:"1px solid #30363d", borderRadius:8, padding:16 }}>
                <div style={{ fontSize:11, color:"#8b949e", marginBottom:10, textTransform:"uppercase", letterSpacing:1 }}>Parameters</div>
                {(algo==="kmeans"||algo==="hierarchical") && (
                  <div style={{ marginBottom:12 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, marginBottom:4 }}><span style={{color:"#8b949e"}}>k (clusters)</span><span style={{color:info.color}}>{k}</span></div>
                    <input type="range" min="2" max="8" value={k} onChange={e=>setK(+e.target.value)} style={{ width:"100%", accentColor:info.color }}/>
                  </div>
                )}
                {algo==="dbscan" && (<>
                  <div style={{ marginBottom:12 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, marginBottom:4 }}><span style={{color:"#8b949e"}}>ε (eps)</span><span style={{color:info.color}}>{eps}</span></div>
                    <input type="range" min="15" max="100" value={eps} onChange={e=>setEps(+e.target.value)} style={{ width:"100%", accentColor:info.color }}/>
                  </div>
                  <div style={{ marginBottom:4 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, marginBottom:4 }}><span style={{color:"#8b949e"}}>minPts</span><span style={{color:info.color}}>{minPts}</span></div>
                    <input type="range" min="2" max="15" value={minPts} onChange={e=>setMinPts(+e.target.value)} style={{ width:"100%", accentColor:info.color }}/>
                  </div>
                </>)}
                {algo==="pca" && <div style={{ fontSize:12, color:"#8b949e", padding:"8px 0" }}>PCA auto-computes eigenvectors — no params needed!</div>}
              </div>

              {/* Run */}
              <button onClick={run} disabled={running} style={{ background: running?"#21262d":`linear-gradient(135deg, ${info.color}, ${info.color}99)`, border:"none", color: running?"#8b949e":"#0d1117", padding:"12px", borderRadius:8, cursor: running?"not-allowed":"pointer", fontSize:14, fontWeight:700, transition:"all 0.3s" }}>
                {running ? "⏳ Running..." : `▶ Run ${info.name}`}
              </button>

              {/* Stats */}
              {stats && (
                <div style={{ background:"#161b22", border:`1px solid ${info.color}44`, borderRadius:8, padding:16 }}>
                  <div style={{ fontSize:11, color:info.color, marginBottom:10, textTransform:"uppercase", letterSpacing:1 }}>Results</div>
                  <div style={{ fontSize:12, color:"#8b949e" }}>Points: <span style={{color:"#e6edf3"}}>{stats.points}</span></div>
                  <div style={{ fontSize:12, color:"#8b949e" }}>Clusters: <span style={{color:info.color}}>{stats.clusters}</span></div>
                  {stats.noise > 0 && <div style={{ fontSize:12, color:"#8b949e" }}>Noise: <span style={{color:"#ff6b6b"}}>{stats.noise}</span></div>}
                  {algo==="pca" && result?.variance && (
                    <div style={{ marginTop:8 }}>
                      <div style={{ fontSize:12, color:"#8b949e" }}>PC1 variance: <span style={{color:info.color}}>{result.variance[0].toFixed(1)}%</span></div>
                      <div style={{ fontSize:12, color:"#8b949e" }}>PC2 variance: <span style={{color:info.color}}>{result.variance[1].toFixed(1)}%</span></div>
                    </div>
                  )}
                  <div style={{ marginTop:10 }}>
                    {stats.sizes.map((s,i)=>(
                      <div key={i} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:3 }}>
                        <div style={{ width:10, height:10, borderRadius:"50%", background:PALETTE[i%PALETTE.length] }}/>
                        <span style={{ fontSize:11, color:"#8b949e" }}>C{i}: {s} pts</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Canvas */}
            <div style={{ background:"#161b22", border:"1px solid #30363d", borderRadius:8, overflow:"hidden" }}>
              <div style={{ padding:"12px 16px", borderBottom:"1px solid #30363d", display:"flex", alignItems:"center", gap:8 }}>
                <span style={{ background:info.color+"33", color:info.color, fontSize:11, padding:"2px 8px", borderRadius:4, fontWeight:700 }}>{info.badge}</span>
                <span style={{ fontSize:13, color:"#e6edf3" }}>{info.name}</span>
                <span style={{ fontSize:12, color:"#8b949e", marginLeft:"auto" }}>{info.complexity}</span>
              </div>
              <svg width="100%" viewBox="0 0 540 440" style={{ display:"block", background:"#0d1117" }}>
                {/* Grid */}
                {[100,200,300,400].map(x=><line key={`gx${x}`} x1={x} y1={0} x2={x} y2={440} stroke="#161b22" strokeWidth={1}/>)}
                {[100,200,300,400].map(y=><line key={`gy${y}`} x1={0} y1={y} x2={540} y2={y} stroke="#161b22" strokeWidth={1}/>)}

                {/* Points */}
                {vizPoints.map((p, i) => {
                  const lbl = labels ? labels[i] : -1;
                  const color = lbl === -2 ? NOISE_COLOR : lbl >= 0 ? PALETTE[lbl % PALETTE.length] : "#4a5568";
                  return (
                    <circle key={i} cx={p[0]} cy={p[1]} r={lbl===-2?3:4} fill={color} opacity={lbl===-2?0.4:0.85}
                      style={{ transition:"all 0.5s ease", filter: lbl>=0 ? `drop-shadow(0 0 3px ${color}66)` : "none" }}/>
                  );
                })}

                {/* Centroids for K-Means */}
                {algo==="kmeans" && result?.centroids?.map((c,i)=>(
                  <g key={i}>
                    <circle cx={c[0]} cy={c[1]} r={10} fill="none" stroke={PALETTE[i%PALETTE.length]} strokeWidth={2} opacity={0.8}/>
                    <text x={c[0]} y={c[1]+4} textAnchor="middle" fontSize={10} fill={PALETTE[i%PALETTE.length]} fontWeight="bold">✕</text>
                  </g>
                ))}

                {/* PCA axes */}
                {algo==="pca" && result && (
                  <g>
                    <text x={10} y={20} fontSize={11} fill="#a78bfa">PC1: {result.variance[0].toFixed(1)}%</text>
                    <text x={10} y={36} fontSize={11} fill="#38bdf8">PC2: {result.variance[1].toFixed(1)}%</text>
                  </g>
                )}

                {!result && (
                  <text x={270} y={230} textAnchor="middle" fontSize={13} fill="#30363d">Click ▶ Run to cluster</text>
                )}
              </svg>
            </div>
          </div>
        )}

        {tab === "code" && (
          <div>
            <div style={{ display:"flex", gap:8, marginBottom:16, flexWrap:"wrap" }}>
              {Object.entries(algoInfo).map(([id,{name,color}])=>(
                <button key={id} onClick={()=>setAlgo(id)} style={{ background: algo===id?color+"22":"#161b22", border:`1px solid ${algo===id?color:"#30363d"}`, color: algo===id?color:"#8b949e", padding:"6px 16px", borderRadius:6, cursor:"pointer", fontSize:13 }}>{name}</button>
              ))}
            </div>
            <div style={{ background:"#161b22", border:"1px solid #30363d", borderRadius:8, overflow:"hidden" }}>
              <div style={{ padding:"10px 16px", borderBottom:"1px solid #30363d", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                <span style={{ fontSize:13, color:"#8b949e" }}>📄 {algo}_clustering.py</span>
                <span style={{ fontSize:11, color:"#30363d" }}>Python 3.10+</span>
              </div>
              <pre style={{ margin:0, padding:20, overflowX:"auto", fontSize:12.5, lineHeight:1.7, color:"#e6edf3", background:"#0d1117" }}>
                <code>{codeSnippets[algo]}</code>
              </pre>
            </div>
          </div>
        )}

        {tab === "about" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
            {Object.entries(algoInfo).map(([id,{name,badge,color,desc,complexity,params}])=>(
              <div key={id} style={{ background:"#161b22", border:`1px solid ${color}33`, borderRadius:8, padding:20 }}>
                <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
                  <span style={{ background:color+"22", color, padding:"3px 10px", borderRadius:4, fontSize:11, fontWeight:700 }}>{badge}</span>
                  <span style={{ fontSize:16, fontWeight:700, color:"#e6edf3" }}>{name}</span>
                  <span style={{ marginLeft:"auto", fontSize:11, color:"#8b949e", background:"#0d1117", padding:"2px 8px", borderRadius:4 }}>{complexity}</span>
                </div>
                <p style={{ fontSize:13, color:"#8b949e", lineHeight:1.6, marginBottom:12 }}>{desc}</p>
                <div style={{ borderTop:`1px solid ${color}22`, paddingTop:12 }}>
                  <div style={{ fontSize:11, color:color, marginBottom:6, textTransform:"uppercase", letterSpacing:1 }}>Key Concepts</div>
                  {params.map((p,i)=>(
                    <div key={i} style={{ fontSize:12, color:"#8b949e", marginBottom:4 }}>
                      <span style={{ color }}>→</span> {p}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

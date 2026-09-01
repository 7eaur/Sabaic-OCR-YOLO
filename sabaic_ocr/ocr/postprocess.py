from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Sequence


@dataclass
class OCRDetection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self): return (self.x1+self.x2)/2.0
    @property
    def cy(self): return (self.y1+self.y2)/2.0
    @property
    def height(self): return max(1e-9,self.y2-self.y1)


def _cluster_lines(detections: Sequence[OCRDetection], tolerance_factor: float=0.60) -> List[List[OCRDetection]]:
    if not detections: return []
    tolerance=max(1e-6, median([d.height for d in detections])*tolerance_factor)
    lines=[]; centers=[]
    for det in sorted(detections,key=lambda d:d.cy):
        if not lines:
            lines.append([det]); centers.append(det.cy); continue
        distances=[abs(det.cy-c) for c in centers]
        best=min(range(len(distances)),key=distances.__getitem__)
        if distances[best]<=tolerance:
            lines[best].append(det); centers[best]=sum(x.cy for x in lines[best])/len(lines[best])
        else:
            lines.append([det]); centers.append(det.cy)
    order=sorted(range(len(lines)),key=lambda i:centers[i])
    return [lines[i] for i in order]


def detections_to_text(detections: Sequence[OCRDetection], id_to_char: Dict[int,str], reading_direction: str="rtl", line_separator: str="\n") -> str:
    if reading_direction not in {"rtl","ltr"}: raise ValueError("reading_direction must be 'rtl' or 'ltr'")
    rendered=[]
    for line in _cluster_lines(detections):
        ordered=sorted(line,key=lambda d:d.cx,reverse=(reading_direction=="rtl"))
        rendered.append("".join(id_to_char[d.class_id] for d in ordered if d.class_id in id_to_char))
    return line_separator.join(rendered)


def tensor_detections_to_objects(tensor):
    out=[]
    for row in tensor:
        v=row.tolist() if hasattr(row,"tolist") else list(row)
        out.append(OCRDetection(class_id=int(v[5]),score=float(v[4]),x1=float(v[0]),y1=float(v[1]),x2=float(v[2]),y2=float(v[3])))
    return out

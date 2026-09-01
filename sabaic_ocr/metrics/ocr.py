from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class EditCounts:
    matches:int=0; substitutions:int=0; deletions:int=0; insertions:int=0
    @property
    def errors(self): return self.substitutions+self.deletions+self.insertions
    def __add__(self,other):
        return EditCounts(self.matches+other.matches,self.substitutions+other.substitutions,self.deletions+other.deletions,self.insertions+other.insertions)


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    n,m=len(reference),len(hypothesis)
    dp=[[0]*(m+1) for _ in range(n+1)]; op=[[""]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=i; op[i][0]="D"
    for j in range(1,m+1): dp[0][j]=j; op[0][j]="I"
    for i in range(1,n+1):
        for j in range(1,m+1):
            if reference[i-1]==hypothesis[j-1]: dp[i][j]=dp[i-1][j-1]; op[i][j]="M"
            else:
                dp[i][j],op[i][j]=min([(dp[i-1][j-1]+1,"S"),(dp[i-1][j]+1,"D"),(dp[i][j-1]+1,"I")],key=lambda x:x[0])
    c=EditCounts(); i,j=n,m
    while i>0 or j>0:
        a=op[i][j]
        if a=="M": c.matches+=1; i-=1; j-=1
        elif a=="S": c.substitutions+=1; i-=1; j-=1
        elif a=="D": c.deletions+=1; i-=1
        elif a=="I": c.insertions+=1; j-=1
        else: break
    return c


def normalize_ocr_text(text): return "".join(text.split())
def words_from_sabaic(text,separator="𐩽"):
    text=text.replace("\n",separator)
    return [w for w in text.split(separator) if w]


def evaluate_pair(reference,hypothesis,separator="𐩽"):
    rc=list(normalize_ocr_text(reference)); hc=list(normalize_ocr_text(hypothesis)); cc=edit_counts(rc,hc)
    rw=words_from_sabaic(reference,separator); hw=words_from_sabaic(hypothesis,separator); wc=edit_counts(rw,hw)
    cer=cc.errors/max(1,len(rc)); wer=wc.errors/max(1,len(rw))
    return {"character":{"reference_count":len(rc),"predicted_count":len(hc),"correct":cc.matches,"wrong":cc.errors,"substitutions":cc.substitutions,"deletions":cc.deletions,"insertions":cc.insertions,"cer":cer,"accuracy_from_cer":max(0.0,1.0-cer)},"word":{"reference_count":len(rw),"predicted_count":len(hw),"correct":wc.matches,"wrong":wc.errors,"substitutions":wc.substitutions,"deletions":wc.deletions,"insertions":wc.insertions,"wer":wer,"accuracy_from_wer":max(0.0,1.0-wer)}}


def evaluate_corpus(pairs: Iterable[Tuple[str,str]],separator="𐩽"):
    ct=EditCounts(); wt=EditCounts(); rct=hct=rwt=hwt=samples=0
    for reference,hypothesis in pairs:
        samples+=1; rc=list(normalize_ocr_text(reference)); hc=list(normalize_ocr_text(hypothesis)); ct=ct+edit_counts(rc,hc); rct+=len(rc); hct+=len(hc)
        rw=words_from_sabaic(reference,separator); hw=words_from_sabaic(hypothesis,separator); wt=wt+edit_counts(rw,hw); rwt+=len(rw); hwt+=len(hw)
    cer=ct.errors/max(1,rct); wer=wt.errors/max(1,rwt)
    return {"samples":samples,"character":{"reference_count":rct,"predicted_count":hct,"correct":ct.matches,"wrong":ct.errors,"substitutions":ct.substitutions,"deletions":ct.deletions,"insertions":ct.insertions,"cer":cer,"accuracy_from_cer":max(0.0,1.0-cer)},"word":{"reference_count":rwt,"predicted_count":hwt,"correct":wt.matches,"wrong":wt.errors,"substitutions":wt.substitutions,"deletions":wt.deletions,"insertions":wt.insertions,"wer":wer,"accuracy_from_wer":max(0.0,1.0-wer)}}

#!/usr/bin/env python3
"""
s2s_fusion_v1_3.py — S2S Full Multi-Sensor Fusion Certifier (v1.3)

Fuses up to 5 sensor streams: IMU | EMG | LiDAR | Thermal | PPG

Human-in-Loop Score (0-100) answers: is a real human doing a real interaction?

Coherence checks per pair:
  IMU  <-> EMG     muscle activation co-occurs with motion
  IMU  <-> PPG     motion present + valid pulse = strong human proof
  IMU  <-> Thermal body heat present when human-scale motion exists
  IMU  <-> LiDAR   movement shows in both sensors simultaneously
  EMG  <-> PPG     two biological signals simultaneously = strong human proof
  EMG  <-> Thermal active muscle generates heat
  PPG  <-> Thermal both confirm body temperature / human presence
  LiDAR<-> Thermal scene geometry + heat source confirmation

Score breakdown:
  40 pts: stream quality (biologically-weighted)
  40 pts: pairwise coherence
  20 pts: biological signal bonuses (pulse, bursts, body heat)
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple

FUSION_MIN_OVERLAP_FRACTION = 0.70
FUSION_MIN_STREAMS = 2
SENSOR_WEIGHTS = {"IMU": 1.0, "EMG": 2.0, "PPG": 2.0, "THERMAL": 1.5, "LIDAR": 0.8}
TIER_SCORE = {"GOLD": 100, "SILVER": 70, "BRONZE": 40, "REJECTED": 0}


def _ov(sa: Dict, sb: Dict) -> float:
    a0 = sa.get("frame_start_ts_ns") or 0; a1 = sa.get("frame_end_ts_ns") or 0
    b0 = sb.get("frame_start_ts_ns") or 0; b1 = sb.get("frame_end_ts_ns") or 0
    if a1 <= a0 or b1 <= b0: return 0.0
    ov0 = max(a0, b0); ov1 = min(a1, b1)
    if ov1 <= ov0: return 0.0
    return (ov1 - ov0) / max(min(a1-a0, b1-b0), 1)

def _rank(t: str) -> int: return {"GOLD":3,"SILVER":2,"BRONZE":1,"REJECTED":0}.get(t,0)
def _synth(c: Dict) -> bool: return "SUSPECT_SYNTHETIC" in c.get("flags",[])
def _pulse(ppg: Dict) -> bool:
    return any(ch.get("has_pulse",False) for ch in (ppg.get("per_channel") or {}).values())


# --- per-pair coherence functions ---

def _imu_emg(imu, emg):
    d = {"pair":"IMU_EMG","type":"biological_motion","temporal_overlap": round(_ov(imu,emg),4)}
    if _ov(imu,emg)<0.60: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(imu): d["fail"]="IMU_SYNTHETIC"; return False,0,d
    if _synth(emg): d["fail"]="EMG_SYNTHETIC"; return False,0,d
    if _rank(imu.get("tier","REJECTED"))<1 or _rank(emg.get("tier","REJECTED"))<1:
        d["fail"]="ONE_STREAM_REJECTED"; return False,0,d
    cv = (imu.get("metrics") or {}).get("cv"); bf = (emg.get("notes") or {}).get("mean_burst_frac",0)
    mot = cv is not None and 1e-6<cv<1.0; burst = bf>=0.05
    d.update({"imu_has_motion":mot,"emg_burst_frac":round(bf,4),"imu_cv":round(cv,6) if cv else None})
    if mot and burst:   d["coherence"]="MOTION_AND_MUSCLE_CONFIRMED"; conf=95
    elif mot or burst:  d["note"]="ONE_SIDE_ACTIVE";                  conf=65
    else:               d["warn"]="BOTH_IDLE";                        conf=40
    d["confidence"]=conf; return conf>=40,conf,d

def _imu_ppg(imu, ppg):
    d = {"pair":"IMU_PPG","type":"motion_vitals","temporal_overlap":round(_ov(imu,ppg),4)}
    if _ov(imu,ppg)<0.60: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(imu): d["fail"]="IMU_SYNTHETIC"; return False,0,d
    if _synth(ppg): d["fail"]="PPG_SYNTHETIC"; return False,0,d
    pulse=_pulse(ppg); hr=(ppg.get("vitals") or {}).get("heart_rate_bpm")
    cv=(imu.get("metrics") or {}).get("cv"); mot=cv is not None and 1e-6<cv<1.0
    hr_ok=hr is not None and 30<=hr<=220
    d.update({"ppg_has_pulse":pulse,"ppg_hr_bpm":hr,"imu_has_motion":mot,"hr_valid":hr_ok})
    if not pulse:  d["fail"]="NO_PULSE";        return False,0,d
    if not hr_ok:  d["fail"]="HR_OUT_OF_RANGE"; return False,0,d
    conf=min(85+(10 if mot else 0),100)
    d["coherence"]="PULSE_AND_MOTION_CONFIRMED" if mot else "PULSE_CONFIRMED_AT_REST"
    d["confidence"]=conf; return True,conf,d

def _emg_ppg(emg, ppg):
    d = {"pair":"EMG_PPG","type":"dual_biological","temporal_overlap":round(_ov(emg,ppg),4)}
    if _ov(emg,ppg)<0.60: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(emg): d["fail"]="EMG_SYNTHETIC"; return False,0,d
    if _synth(ppg): d["fail"]="PPG_SYNTHETIC"; return False,0,d
    bf=(emg.get("notes") or {}).get("mean_burst_frac",0); pulse=_pulse(ppg)
    hrv=(ppg.get("vitals") or {}).get("hrv_rmssd_ms",0); hrv_ok=hrv>3.0
    d.update({"emg_burst_frac":round(bf,4),"ppg_has_pulse":pulse,"hrv_rmssd_ms":hrv,"hrv_ok":hrv_ok})
    if not pulse: d["fail"]="NO_PULSE"; return False,0,d
    if bf>=0.03 and pulse and hrv_ok:   d["coherence"]="DUAL_BIOLOGICAL_CONFIRMED"; conf=95
    elif pulse and hrv_ok:              d["coherence"]="PPG_HRV_CONFIRMED";         conf=80
    elif pulse:                         d["warn"]="PULSE_LOW_HRV";                  conf=60
    else: conf=30
    d["confidence"]=conf; return conf>=40,conf,d

def _imu_thermal(imu, thermal):
    d = {"pair":"IMU_THERMAL","type":"motion_heatsignature","temporal_overlap":round(_ov(imu,thermal),4)}
    if _ov(imu,thermal)<0.40: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(imu): d["fail"]="IMU_SYNTHETIC"; return False,0,d
    if _synth(thermal): d["fail"]="THERMAL_SYNTHETIC"; return False,0,d
    human=(thermal.get("human_presence") or {}).get("human_present",False)
    bf=(thermal.get("human_presence") or {}).get("body_heat_fraction",0)
    cv=(imu.get("metrics") or {}).get("cv"); mot=cv is not None and 1e-6<cv<1.0
    d.update({"thermal_human":human,"body_frac":round(bf,4),"imu_motion":mot})
    if human and mot:  d["coherence"]="BODY_HEAT_AND_MOTION_CONFIRMED"; conf=90
    elif human:        d["coherence"]="BODY_HEAT_CONFIRMED";            conf=75
    else:              d["warn"]="NO_BODY_HEAT";                        conf=30
    d["confidence"]=conf; return conf>=30,conf,d

def _imu_lidar(imu, lidar):
    d = {"pair":"IMU_LIDAR","type":"motion_scene","temporal_overlap":round(_ov(imu,lidar),4)}
    if _ov(imu,lidar)<0.60: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(imu):   d["fail"]="IMU_SYNTHETIC";   return False,0,d
    if _synth(lidar): d["fail"]="LIDAR_SYNTHETIC";  return False,0,d
    cv=(imu.get("metrics") or {}).get("cv"); mot=cv is not None and 1e-6<cv<1.0
    lmot=(lidar.get("motion") or {}).get("micro_motion_confirmed",False)
    d.update({"imu_motion":mot,"lidar_motion":lmot})
    if mot and lmot: d["coherence"]="MOTION_BOTH_SENSORS"; conf=90
    elif mot or lmot: d["note"]="MOTION_ONE_SENSOR";       conf=65
    else:             d["note"]="BOTH_STATIC";             conf=45
    d["confidence"]=conf; return True,conf,d

def _emg_thermal(emg, thermal):
    d = {"pair":"EMG_THERMAL","type":"muscle_heat","temporal_overlap":round(_ov(emg,thermal),4)}
    if _ov(emg,thermal)<0.40: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(emg):     d["fail"]="EMG_SYNTHETIC";     return False,0,d
    if _synth(thermal): d["fail"]="THERMAL_SYNTHETIC"; return False,0,d
    bf=(emg.get("notes") or {}).get("mean_burst_frac",0)
    human=(thermal.get("human_presence") or {}).get("human_present",False)
    d.update({"emg_burst_frac":round(bf,4),"thermal_human":human})
    if human and bf>=0.03: d["coherence"]="MUSCLE_HEAT_CONFIRMED"; conf=85
    elif human:            d["note"]="BODY_HEAT_EMG_IDLE";         conf=65
    elif bf>=0.03:         d["warn"]="EMG_ACTIVE_NO_THERMAL";      conf=40
    else: conf=35
    d["confidence"]=conf; return conf>=30,conf,d

def _ppg_thermal(ppg, thermal):
    d = {"pair":"PPG_THERMAL","type":"dual_heatsignature","temporal_overlap":round(_ov(ppg,thermal),4)}
    if _ov(ppg,thermal)<0.30: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(ppg):     d["fail"]="PPG_SYNTHETIC";     return False,0,d
    if _synth(thermal): d["fail"]="THERMAL_SYNTHETIC"; return False,0,d
    pulse=_pulse(ppg); hr=(ppg.get("vitals") or {}).get("heart_rate_bpm")
    human=(thermal.get("human_presence") or {}).get("human_present",False)
    bf=(thermal.get("human_presence") or {}).get("body_heat_fraction",0)
    d.update({"ppg_pulse":pulse,"ppg_hr":hr,"thermal_human":human,"body_frac":round(bf,4)})
    if pulse and human: d["coherence"]="PULSE_AND_BODY_HEAT_CONFIRMED"; conf=92
    elif pulse:         d["note"]="PULSE_OK_THERMAL_NOT_AIMED";         conf=70
    elif human:         d["note"]="BODY_HEAT_OK_PPG_NOT_ON_SKIN";       conf=60
    else: d["fail"]="NEITHER_BIOLOGICAL"; return False,0,d
    d["confidence"]=conf; return True,conf,d

def _lidar_thermal(lidar, thermal):
    d = {"pair":"LIDAR_THERMAL","type":"scene_heatsource","temporal_overlap":round(_ov(lidar,thermal),4)}
    if _ov(lidar,thermal)<0.40: d["fail"]="INSUFFICIENT_OVERLAP"; return False,0,d
    if _synth(lidar):   d["fail"]="LIDAR_SYNTHETIC";   return False,0,d
    if _synth(thermal): d["fail"]="THERMAL_SYNTHETIC"; return False,0,d
    human=(thermal.get("human_presence") or {}).get("human_present",False)
    d.update({"thermal_human":human,"lidar_tier":lidar.get("tier")})
    conf=80 if human else 50
    if human: d["coherence"]="SCENE_AND_HEAT_CONFIRMED"
    d["confidence"]=conf; return True,conf,d

def _generic(sa, sb, na, nb):
    d = {"pair":f"{na}_{nb}","type":"temporal_only","temporal_overlap":round(_ov(sa,sb),4)}
    if _synth(sa): d["fail"]=f"{na}_SYNTHETIC"; return False,0,d
    if _synth(sb): d["fail"]=f"{nb}_SYNTHETIC"; return False,0,d
    ov=_ov(sa,sb); conf=int(ov*80)
    if ov<FUSION_MIN_OVERLAP_FRACTION: d["warn"]="LOW_OVERLAP"
    d["confidence"]=conf; return ov>=FUSION_MIN_OVERLAP_FRACTION,conf,d

def _route(sa: Dict, sb: Dict) -> Tuple[bool,int,Dict]:
    ta=sa.get("_sensor_type","").upper(); tb=sb.get("_sensor_type","").upper()
    p=frozenset([ta,tb])
    g=lambda t,s: s if s.get("_sensor_type","").upper()==t else (sa if sa.get("_sensor_type","").upper()==t else sb)
    if p=={"IMU","EMG"}:      return _imu_emg(g("IMU",sa),g("EMG",sb))
    if p=={"IMU","PPG"}:      return _imu_ppg(g("IMU",sa),g("PPG",sb))
    if p=={"IMU","THERMAL"}:  return _imu_thermal(g("IMU",sa),g("THERMAL",sb))
    if p=={"IMU","LIDAR"}:    return _imu_lidar(g("IMU",sa),g("LIDAR",sb))
    if p=={"EMG","PPG"}:      return _emg_ppg(g("EMG",sa),g("PPG",sb))
    if p=={"EMG","THERMAL"}:  return _emg_thermal(g("EMG",sa),g("THERMAL",sb))
    if p=={"PPG","THERMAL"}:  return _ppg_thermal(g("PPG",sa),g("THERMAL",sb))
    if p=={"LIDAR","THERMAL"}:return _lidar_thermal(g("LIDAR",sa),g("THERMAL",sb))
    return _generic(sa,sb,ta,tb)


def _hil(streams, coh, flags):
    if any("SYNTHETIC" in f for f in flags): return 0
    tw=sum(SENSOR_WEIGHTS.get(s.get("_sensor_type",""),1.0) for s in streams)
    sp=sum((TIER_SCORE.get(s.get("tier","REJECTED"),0)/100.0)*SENSOR_WEIGHTS.get(s.get("_sensor_type",""),1.0) for s in streams)
    stream_pts=(sp/max(tw,1.0))*40
    confs=[c for(_,c,_) in coh.values()]
    coh_pts=((sum(confs)/len(confs))/100.0*40) if confs else 0.0
    bio=0.0
    for s in streams:
        t=s.get("_sensor_type","").upper()
        if t=="EMG" and (s.get("notes") or {}).get("mean_burst_frac",0)>=0.05: bio+=7
        elif t=="PPG":
            if _pulse(s): bio+=7
            if (s.get("vitals") or {}).get("hrv_rmssd_ms",0)>5.0: bio+=3
        elif t=="THERMAL" and (s.get("human_presence") or {}).get("human_present",False): bio+=5
    return max(0,min(100,int(stream_pts+coh_pts+min(bio,20))))


def _tier(valid, score, all_ok):
    if not valid: return "REJECTED"
    ranks=[_rank(s.get("tier","REJECTED")) for s in valid]
    mr=min(ranks); ar=sum(ranks)/len(ranks)
    if mr>=2 and ar>=2.5 and score>=70 and all_ok: return "GOLD"
    if mr>=1 and ar>=1.5 and score>=45:            return "SILVER"
    if mr>=1 and score>=20:                        return "BRONZE"
    return "REJECTED"


class FusionCertifier:
    """
    Full 5-sensor fusion certifier.

    fc = FusionCertifier(device_id="glove_v2", session_id="s001")
    fc.add_imu_cert(imu_cert)
    fc.add_emg_cert(emg_cert)
    fc.add_lidar_cert(lidar_cert)
    fc.add_thermal_cert(thermal_cert)
    fc.add_ppg_cert(ppg_cert)
    result = fc.certify()
    print(result["human_in_loop_score"])
    """
    def __init__(self, device_id="unknown", session_id=None, **kw):
        self.device_id=device_id; self.session_id=session_id; self._streams: List[Dict]=[]

    def add_stream(self, sid, cert, sensor_type=None):
        e=dict(cert); e["_stream_id"]=sid
        e["_sensor_type"]=(sensor_type or cert.get("sensor_type","UNKNOWN")).upper()
        self._streams.append(e); return self

    def add_imu_cert(self,c):     return self.add_stream("imu",    c,"IMU")
    def add_emg_cert(self,c):     return self.add_stream("emg",    c,"EMG")
    def add_lidar_cert(self,c):   return self.add_stream("lidar",  c,"LIDAR")
    def add_thermal_cert(self,c): return self.add_stream("thermal",c,"THERMAL")
    def add_ppg_cert(self,c):     return self.add_stream("ppg",    c,"PPG")

    def certify(self) -> Dict:
        flags: List[str]=[]; n=len(self._streams)
        if n<FUSION_MIN_STREAMS:
            return {"status":"FAIL","tier":"REJECTED","sensor_type":"FUSION",
                    "human_in_loop_score":0,"flags":["INSUFFICIENT_STREAMS"],
                    "notes":{"required":FUSION_MIN_STREAMS,"provided":n},
                    "tool":"s2s_fusion_v1_3","issued_at_ns":time.time_ns()}

        for s in self._streams:
            if _synth(s): flags.append(f"STREAM_{s['_stream_id'].upper()}_SUSPECT_SYNTHETIC")

        coh_r: Dict[str,Tuple[bool,int,Dict]]={};  coh_d: Dict[str,Any]={}; failed=0
        for i in range(n):
            for j in range(i+1,n):
                sa,sb=self._streams[i],self._streams[j]
                key=f"{sa['_stream_id']}_vs_{sb['_stream_id']}"
                ok,conf,det=_route(sa,sb)
                coh_r[key]=(ok,conf,det); coh_d[key]=det
                if not ok:
                    failed+=1
                    flags.append(f"INCOHERENT_{key.upper()}" + (f":{det['fail']}" if 'fail' in det else ""))

        if any("SUSPECT_SYNTHETIC" in f for f in flags):
            flags.append("FUSION_REJECTED_SYNTHETIC_STREAM")
            return {"status":"FAIL","tier":"REJECTED","sensor_type":"FUSION",
                    "human_in_loop_score":0,"n_streams":n,
                    "streams":self._sum(),"coherence_checks":coh_d,
                    "flags":list(dict.fromkeys(flags)),
                    "device_id":self.device_id,"session_id":self.session_id,
                    "tool":"s2s_fusion_v1_3","issued_at_ns":time.time_ns()}

        all_ok=failed==0
        score=_hil(self._streams,coh_r,flags)
        valid=[s for s in self._streams if s.get("tier")!="REJECTED"]
        t=_tier(valid,score,all_ok)
        total_p=n*(n-1)//2
        starts=[s.get("frame_start_ts_ns") for s in self._streams if s.get("frame_start_ts_ns")]
        ends=[s.get("frame_end_ts_ns") for s in self._streams if s.get("frame_end_ts_ns")]
        f0=min(starts) if starts else None; f1=max(ends) if ends else None
        return {
            "status":              "PASS" if t!="REJECTED" else "FAIL",
            "tier":                t,
            "sensor_type":         "FUSION",
            "human_in_loop_score": score,
            "n_streams":           n,
            "n_valid_streams":     len(valid),
            "fusion_start_ts_ns":  f0,
            "fusion_end_ts_ns":    f1,
            "fusion_duration_ms":  round((f1-f0)/1e6,2) if (f0 and f1) else None,
            "streams":             self._sum(),
            "coherence_checks":    coh_d,
            "flags":               list(dict.fromkeys(flags)),
            "notes": {"total_pairs_checked":total_p,"coherent_pairs":total_p-failed,
                      "failed_pairs":failed,"n_valid_streams":len(valid),
                      "stream_tiers":{s["_stream_id"]:s.get("tier") for s in self._streams}},
            "device_id":    self.device_id,
            "session_id":   self.session_id,
            "tool":         "s2s_fusion_v1_3",
            "issued_at_ns": time.time_ns(),
        }

    def _sum(self):
        return [{"stream_id":s["_stream_id"],"sensor_type":s["_sensor_type"],
                 "tier":s.get("tier","UNKNOWN"),"flags":s.get("flags",[]),
                 "duration_ms":s.get("duration_ms")} for s in self._streams]

    def reset(self): self._streams.clear()

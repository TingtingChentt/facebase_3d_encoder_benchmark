#!/usr/bin/env python3
"""
Build a label that uses the FB-TJ0 `Clinical Diagnosis` column first and fills
it from the `Syndrome` column where it is blank, after checking the two columns
against each other wherever both are filled.


WHY A CHECK IS NEEDED BEFORE FILLING. `Clinical Diagnosis` is free text (812
distinct strings over 2,982 subjects): "CdLS", "Marfans", "NF1",
"Neurofibromotosis Type 1" and "Neurofibromatosis, type 1" are all one
condition. `Syndrome` is the curated vocabulary (filled for all 5,800
subjects). A filled-in row can only carry a `Syndrome` name, so the two columns
must share one vocabulary or the same disorder becomes two classes depending
on which column happened to be filled. Every subject with both columns is
therefore classified:

  agree_exact     same string after case/punctuation/"syndrome" normalisation
  agree_alias     same disorder: abbreviation, misspelling, synonym, subtype
                  ("MPS II" vs "Mucopolysaccharidosis"), or "possible X"
  cd_nonspecific  Clinical Diagnosis names no disorder ("Chromosome
                  abnormality", "copy gains", a list of findings) while
                  Syndrome does
  conflict        the two columns name different disorders/loci
  suspect_syndrome  Syndrome = Moebius but Clinical Diagnosis describes a CNV
                  or another condition (see MOEBIUS note below)
  nondiag         either column is a control / relative / unknown label

LABEL RULE (defaults; each one is recorded per subject in `label_source` so it
can be changed without re-reading this script):
  CD blank                     -> Syndrome                ("syndrome_fill")
  agree_exact / agree_alias    -> Syndrome spelling of the CD disorder
                                  ("clinical"). Same disorder, so using the
                                  curated spelling is what makes CD and
                                  filled rows land in one class.
  cd_nonspecific               -> Syndrome ("syndrome_fill_nonspecific_cd"):
                                  CD carries no diagnosis to use.
  conflict / suspect_syndrome  -> no label (excluded, listed for review)
  nondiag                      -> no label

MOEBIUS. 34 subjects carry Syndrome = "Moebius Syndrome", all with Syndrome
Category "Uncategorized" and OMIM 157900. Only 4 have a Moebius clinical
diagnosis; 18 have CNV/chromosomal text ("copy gains", "Chromosome
abnormality", "22Q", "Type II Collagenopathy"), and several CD-blank ones have
unrelated molecular findings (8p23.2p22 deletion, add(4q), ALG13). The value
looks like a placeholder for "uncategorised chromosomal", not a Moebius
diagnosis. Only the 4 CD-confirmed subjects keep the Moebius label.

Writes:
  data/labels/build/clinical_filled_subjects.csv  one row per subject (audit)
  data/labels/build/clinical_filled_manifest.csv  one row per processed scan
Pool = syndrome_manifest.csv (all processed FB-TJ0 scans) with an existing
.npy, the same pool as the s19/n50 manifests. No class-size filter is applied
here; filtering is the training script's decision.
"""
import os
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS = PROJECT_ROOT / "data/labels"          # manual review table (shipped)
PROCESSED = LABELS / "build"                     # intermediate outputs
PROCESSED.mkdir(parents=True, exist_ok=True)
SRC = PROCESSED / "fbtj0_cohort.csv"   # FB-TJ0 scan list + `Syndrome Category`, written by
                                     # facebench/preprocessing/build_manifests.py
OUT_SUBJ = PROCESSED / "clinical_filled_subjects.csv"
OUT = PROCESSED / "clinical_filled_manifest.csv"
META = (Path(os.environ.get("FACEBASE_ROOT", "FACEBASE_ROOT_not_set"))
        / "FB-TJ0_files/FB00000861_metadata_2020-09-14.csv")

# Syndrome values that are themselves descriptive; a matching descriptive CD
# ("VPI", "Growth deficiency") agrees with them rather than being non-specific.
GENERIC_SY = {"Chromosomal Abnormality", "Growth Delay",
              "Velopharyngeal Insufficiency"}

NONDIAG_SY = {"Relative", "Unaffected Unrelated", "Unknown",
              "Facial Dysmorphia, possibly genetic"}

# CD strings that name no disorder. Matched case-insensitively against the
# whole CD value; only consulted when the Syndrome column is a diagnosis.
NONSPECIFIC_RE = re.compile(
    r"^(chromosom\w*\s*(abnormality|anomal\w*|anomalies|deletion|copy|"
    r"microdeletion.*|termination|7 deletion)|"
    r"autosomal anomalies|abnormal cma|copy (gains?|loss)( and losses)?|"
    r"copy number variants|facial dysmorphia, possibly genetic|"
    r"type [12]|unknown|possible autosomal|gene duplication|"
    r"partial trisomy syndrome|x-linked genetic disorder|skeletal dysplasia|"
    r"new skeletal dysplasia|craniosynostosis|hypermobility|hearing loss|"
    r"growth deficiency|vpi|chromosome q deletion|"
    r"microdeletion on chromosome 16|connection tissue disorder)$", re.I)

# CD values that describe findings rather than a diagnosis (exact strings).
NONSPECIFIC_EXACT = {
    "Chromosomal abnormality, developmental delay (IQ 54), ADHD, narrow palate, otherwise nl PE",
    "Partial agenesis corpus callosum, unilateral cleft lip/palate, short stature, bilat inguinal herniae, hindfoot valgus, dvlptl delay",
    "Developmental delay, Sz",
    "Dysmorphic facies, synophrys, strabismus, VSD, bifid uvula, DD, language delay",
    "Dysmorphic facies, poor growth, cryptorchidism, dvlptl delay",
    "Congenital diaphragmatic hernia, scoliosis, PDA, dysmorphic facies",
    "Craniosynostosis, multiple congenital anomalies",
    "Agenesis of the corpus callosum",
    "Chromosomal anomaly, developmental delays, benign familial macrocephaly",
    "Developmental delay and multiple congenital anomalies",
    "Chromosomal Abnormality",
    "Chromosomal anomaly",
    "Chromosome Anomalies",
    "Chromosomal anomalies",
    "Chromosome anomaly",
    "Hypermobility, low tone, scoliosis, pelvic tilt.",
    "Lower mandibular jaw distraction on 12/26/04 and 7/2012",
    "Dilated cardiomyopathy, scoliosis",
    "Indeterminite connective tissue disorder",
    "Siezures, microcephaly",
    "Brain injury, microcephaly, probably CP, electrical seizures",
    "possible neuro muscular disorder, conditions due to anomaly of unspecified chromosome",
    "Chromosome 3 pair - anomaly",
    "autosomal anomalies",
    "Autosomal anomalies",
    "skeletal dysplasia",
    "Chromosome abnormality",
    "Chromosome Abnormality",
    "unknown - broad digits, unusual build, dymorphic, myopia",
}

# (Syndrome, Clinical Diagnosis) pairs naming different disorders or loci.
CONFLICT = {
    ("18p Deletion", "17p deletion"),
    ("18q Deletion", "18p11.32 Deletion"),
    ("1q44 Deletion", "1q21 duplication"),
    ("21q22 Deletion", "Trisomy 22-21 syndrome/Trisomey 22 syndrome,"),
    ("4q31 Deletion", "11q deletion"),
    ("6p25.3 Deletion", "Chromosome 6 long arm deletion"),
    ("22q 11.2 Deletion Syndrome", "Rett Syndrome"),
    ("22q 11.2 Deletion Syndrome", "Deletion of 2p13.2p11.2"),
    ("5p Deletion Cri du Chat", "8p23.11, 8p11.2"),
    ("Fragile X", "NF1"),
    ("Fragile X", "Stickler/Marshall (self-report)"),
    ("Fragile X", "Stickler/Marshall Syndrome"),
    ("Goldenhar Syndrome", "Chromosome p duplication"),
    ("Jacobsen Syndrome", "Feta Alcohol syndrome/copy gains and losses"),
    ("Jacobsen Syndrome", "KHE/Gorham findings"),
    ("Pitt-Hopkins Syndrome", "NRXN1"),
    ("Rubinstein-Taybi Syndrome", "Chromosome 16q deletion"),
    ("Saethre-Chotzen Syndrome", "Pfeiffer Syndrome"),
    ("Trisomy 8", "Trisomy 18 mosaicism"),
    ("Turner Syndrome", "Possible TBX1"),
    ("X-Linked Hypohidrotic Ectodermal Dysplasia", "fibrous dysplasia"),
}


def norm(s):
    s = s.lower()
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(syndrome|sydrome|disease|the)\b", " ", s)
    return " ".join(s.split())


def canon_case(series, weight):
    """Merge case-only spelling variants to the most frequent spelling."""
    key = series.str.lower()
    best = (pd.DataFrame({"k": key, "v": series, "w": weight})
              .groupby(["k", "v"]).w.sum().reset_index()
              .sort_values("w").drop_duplicates("k", keep="last")
              .set_index("k")["v"])
    return key.map(best)


def classify(cd, sy):
    if pd.isna(cd):
        return "cd_missing"
    if sy in NONDIAG_SY or norm(cd) in {"control", "none", "normal",
                                        "non relative control"}:
        return "nondiag"
    if norm(cd) == norm(sy):
        return "agree_exact"
    if sy in GENERIC_SY:          # both columns say the same generic thing
        return "agree_alias"
    if (sy, cd) in CONFLICT:
        return "conflict"
    if sy == "Moebius Syndrome" and "moebius" not in cd.lower():
        return "suspect_syndrome"
    if cd in NONSPECIFIC_EXACT or NONSPECIFIC_RE.match(cd.strip()):
        return "cd_nonspecific"
    return "agree_alias"


def main():
    meta = pd.read_csv(META)
    subj = pd.DataFrame({
        "fbid": meta["FBID"],
        "clinical_diagnosis": meta["Clinical Diagnosis"].str.strip(),
        "syndrome": meta["Syndrome"].str.strip(),
        "syndrome_category": meta["Syndrome Category"],
    })
    subj["syndrome"] = canon_case(subj["syndrome"], 1)
    subj["check"] = [classify(c, s) for c, s in
                     zip(subj.clinical_diagnosis, subj.syndrome)]

    src = {"cd_missing": "syndrome_fill", "agree_exact": "clinical",
           "agree_alias": "clinical",
           "cd_nonspecific": "syndrome_fill_nonspecific_cd",
           "conflict": "excluded_conflict",
           "suspect_syndrome": "excluded_suspect_syndrome",
           "nondiag": "excluded_nondiag"}
    subj["label_source"] = subj["check"].map(src)
    lab = subj["syndrome"].where(~subj.label_source.str.startswith("excluded"))
    # CD blank AND Syndrome non-diagnostic -> nothing to fill with.
    blank_nd = (subj.check == "cd_missing") & subj.syndrome.isin(NONDIAG_SY)
    subj.loc[blank_nd, "label_source"] = "excluded_nondiag"
    lab[blank_nd] = None
    # CD-blank Moebius rows inherit the doubt about the value itself.
    blank_moeb = (subj.check == "cd_missing") & (subj.syndrome == "Moebius Syndrome")
    subj.loc[blank_moeb, "label_source"] = "excluded_suspect_syndrome"
    lab[blank_moeb] = None
    subj["label"] = lab
    subj.to_csv(OUT_SUBJ, index=False)

    scans = pd.read_csv(SRC)
    scans = scans[scans["eligible"]]   # scans in the benchmark (data/splits/scans.csv)
    scans = scans[["scan_id", "src_path", "fbid", "out_path"]].merge(
        subj[["fbid", "clinical_diagnosis", "syndrome", "check",
              "label_source", "label"]], on="fbid", how="left")
    assert scans["syndrome"].notna().all(), "scan with no metadata row"
    scans = scans.rename(columns={"label": "clinical_filled"})
    scans.to_csv(OUT, index=False)
    print(f"subjects: {len(subj)} -> {OUT_SUBJ}")
    print(f"scans:    {len(scans)} -> {OUT}")


if __name__ == "__main__":
    main()

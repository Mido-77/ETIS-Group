# ETIS — Arabic Data Source Citations

For use in Chapter 3.3 (Data Collection Methodology) and the References list.
Style matches Group_Project_Proposal_final.docx (Harvard-style, matching existing refs).

---

## Sources used as training data

**1. Salloum/Gaber bilingual corpus (300 rows: 150 phishing + 150 legitimate)**

Salloum, S., Gaber, T., Vadera, S. and Shaalan, K., 2023. *A New English/Arabic
Parallel Corpus for Phishing Emails*. ACM Transactions on Asian and Low-Resource
Language Information Processing, 22(4), Article 3606031.
Available at: https://doi.org/10.1145/3606031
[Accessed 15 7 2026].

> Note: one of the co-authors (Tarek Gaber) is affiliated with Suez Canal
> University, Egypt — worth mentioning in Chapter 3.3 as a point of relevance,
> not just citing the DOI.

---

**2. Kaggle "Fully Dataset" (1,250 rows: 1,000 legitimate + 250 phishing)**

Hosni, A., n.d. *BERT for Arabic Phishing and Legitimate Emails*. [Online]
Available at: https://www.kaggle.com/code/ahmadhosni/bert-for-arabic-phishing-and-legitimate-emails
[Accessed 15 7 2026].

> Note: this dataset's phishing rows are translated 419/advance-fee-fraud
> scam letters and its legitimate rows are translated from the UCI SMS Spam
> Collection's ham messages — worth disclosing this as a limitation in
> Chapter 3.3 or 7.3 (translated, not organically Egyptian/Arabic-authored
> content).

---

**3. Dialectal Arabic phishing/safe SMS dataset (~333 unique rows after
deduplication of 167 exact duplicates from the original 500)**

Nizar, S., n.d. *Arabic Phishing & Social Engineering Dataset*. [Online]
Available at: https://www.kaggle.com/datasets/sultannizar/arabic-phishing-dataset
[Accessed 15 7 2026].

> Note: unverified with full confidence — the filename matched this dataset
> page, but wasn't independently confirmed against the page's actual file
> listing. Worth a final visual check before submission.

---

**4. GitHub Arabic-Text-Spam-Detection (78 rows: 26 spam + 52 non-spam,
Saudi/Gulf dialect)**

Mukhairez, H., n.d. *Arabic-Text-Spam-Detection*. [Online]
Available at: https://github.com/HosamMukhairez/Arabic-Text-Spam-Detection
[Accessed 15 7 2026].

> Note: also the source of the ~44-word Arabic phishing keyword blacklist
> saved for Phase 2's rule-based engine. No license file in the repo — cite
> as a GitHub repository per above, not as a formally licensed dataset.

---

## Sources cited in Literature Review only (not used as training data)

**5. Arabic SMS Scam detection paper — cite in §2.3 (Academic Research in
Phishing Detection) as evidence of the Arabic-phishing-research gap**

Zawari, A.H., Alsulami, B.S. and Alqurashi, F.A., 2025. *Detecting Arabic
SMS Scam Messages Using a Hybrid Ensemble Machine Learning Algorithms*.
Journal of Information Systems Engineering and Management, 10(55s), pp.148-156.

---

**6. Alhayan et al. deceptive-text paper — cite in §2.3, pairs well with
citation 5 as a second data point on Arabic deception-detection scarcity**

Alhayan, F., Himdi, H.T. and Alharbi, B., 2024. *Unveiling Deception in
Arabic: Optimization of Deceptive Text Detection Across Formal and Informal
Genres*. IEEE Access, 12, pp.94216-94230.

---

## Suggested Chapter 3.3 paragraph (draft, edit as needed)

"Given the scarcity of publicly available Arabic phishing corpora noted in
prior work (Zawari et al., 2025; Alhayan et al., 2024), this study combined
four independently sourced Arabic-language datasets: a peer-reviewed
English/Arabic parallel phishing email corpus (Salloum et al., 2023), a
translated advance-fee-fraud and SMS-ham corpus (Hosni, n.d.), a dialectal
Arabic phishing/safe SMS dataset (Nizar, n.d.), and an authentic
Gulf-dialect spam/ham message set (Mukhairez, n.d.). These were supplemented
with synthetic Egyptian-brand-specific phishing samples (see §3.3.x) to
address the remaining gap in Egyptian-dialect, brand-specific coverage."

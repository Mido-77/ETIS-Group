# Grounding sources for synthetic Egyptian phishing examples

The 54 synthetic examples in `add_synthetic_egyptian_examples.py` are not
generic/imagined phishing content -- each pattern used is grounded in a
real, currently-documented Egyptian fraud tactic reported by the bank/company
itself or by Egyptian news outlets. This list is for citing in Chapter 3.3
(Data Collection Methodology) to justify the synthetic data's realism.

## Patterns used, and their real-world source

**Fake prize/reward message** (used for Banque Misr, NBE, Vodafone examples)
Source: Bank Misr's own customer awareness campaign explicitly quotes and
warns against this exact tactic ("مبروك كسبت جائزة من بنك مصر").
https://www.bnok24.com/2025/05/ (accessed 2026-07-19)

**Fake "sent by mistake, please return it" message** (used for Vodafone Cash,
Talabat refund examples)
Source: documented, actively-reported Vodafone Cash scam pattern, covered by
multiple Egyptian outlets in 2022 and again in 2026.
https://techrevieweg.com/ (accessed 2026-07-19)
https://www.elwatannews.com/news/details/8262292 (accessed 2026-07-19)

**Fake account-update / new-service-activation request** (used for Instapay,
Etisalat/e&, CIB, QNB examples)
Source: Instapay's own published fraud warning describes fraudsters posing as
the app and asking users to "update their account" or "activate a new
service."
https://www.dostor.org/5325790 (accessed 2026-07-19)

**Fake pending-shipment link** (used for Noon, Jumia examples)
Source: named by the Central Bank of Egypt as one of the most common
nationwide fraud tactics in its 2026 awareness campaign.
https://sis.gov.eg/ar/... (accessed 2026-07-19)

**Fake OTP/PIN request via call or SMS** (used across multiple bank/telecom
examples)
Source: universally warned against by every source checked -- CBE, HSBC
Egypt, Vodafone Egypt, Instapay all explicitly state they never request OTP
or PIN by phone/SMS.
https://www.hsbc.com.eg/ar-eg/help/security/fraud-guide/ (accessed 2026-07-19)

**Fake card-blocked message requiring a callback** (used for AAIB example)
Source: this exact pattern (in Arabic, targeting a different Gulf bank) was
also found verbatim in the GitHub Arabic-Text-Spam-Detection dataset (see
docs/arabic_sources_citations.md, source #4) -- a real collected example of
this tactic, adapted here to an Egyptian bank context.

**WhatsApp hacked-contact urgent transfer request**
Source: named explicitly by the Central Bank of Egypt as a common tactic in
its nationwide fraud-awareness campaign.
https://cnnbusinessarabic.com/technology/1138603/... (accessed 2026-07-19)

## Suggested Chapter 3.3 citation sentence

"Synthetic Egyptian brand-impersonation examples were constructed using fraud
patterns explicitly documented by the Central Bank of Egypt's public
awareness campaigns and by Vodafone Egypt, Instapay, and Bank Misr's own
customer fraud warnings, rather than generic or imagined phishing content,
to ensure ecological validity for the Egyptian threat landscape."

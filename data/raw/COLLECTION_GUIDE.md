# ETIS real-message collection guide

Goal: 200+ real messages (target 300), mostly **legitimate**, from real life. Today the dataset
has almost no real legitimate bank / wallet / delivery / OTP messages, so the model learns
"money or account words = phishing". Your messages fix that.

## Rules

1. **Only your own messages**, or messages from people who have said yes. Nothing from groups
   or other people's chats.
2. **Anonymise before you save.** Replace, do not delete the sentence:
   - names -> `[NAME]`
   - phone numbers -> `[PHONE]`
   - card / account / IBAN / national ID numbers -> `XXXX` (keep the last-4 style if the message has one)
   - addresses -> `[ADDRESS]`
   - links containing a personal token or tracking id -> keep the domain, replace the rest with `/[ID]`
   - OTP codes -> a random code of the same length
   - amounts and balances -> a similar made-up number (do not leave real balances)
3. Never paste a message you are unsure about. Skip it.
4. Do not edit the wording otherwise. Typos, dialect and emojis are the point.

## What to collect (aim for this mix of legitimate messages)

| category | examples | target |
|---|---|---|
| bank_notification | debit/credit alerts, card payment, statement ready | 40 |
| otp | codes sent by banks, Instapay, Vodafone Cash, apps, delivery | 40 |
| wallet_payment | Vodafone Cash, Instapay, Fawry, Orange Cash receipts | 30 |
| telecom | balance, bill, package renewal, offers from your operator | 30 |
| delivery | Bosta, Jumia, Noon, Talabat, Amazon updates | 30 |
| government_service | electricity/gas/water bills, e-gov, traffic, tax | 15 |
| personal_work | normal chat/work messages (Arabic, Franco-Arabic, English, mixed) | 30 |
| anti_fraud_warning | real warnings from banks/telecoms about scams | 10 |

Also add **real phishing/scam messages you actually received** (label `phishing` or `scam`),
as many as you have. Set `received_in_real_life = yes`. Do not write new phishing messages.

## File format

Copy `collection_template.csv`, one file per contributor, named `messages_<initials>.csv`.
Columns:

- `text` – the anonymised message (UTF-8, keep line breaks inside quotes)
- `label` – `legitimate`, `phishing`, `scam` or `spam`
- `category` – one value from the table above, or `phishing_received`
- `channel` – `sms`, `whatsapp` or `email`
- `language` – `ar`, `en` or `mixed`
- `brand` – the real brand it comes from or imitates, if any
- `contributor` – your initials
- `received_in_real_life` – `yes` (everything here should be yes)

Save as **CSV UTF-8** (Excel: Save As -> CSV UTF-8). Send the file to Mido, do not commit it
to a public repo.

## Checklist before sending

- [ ] no names, phone numbers, card/account numbers, addresses
- [ ] OTPs and balances replaced
- [ ] every row has label, category, channel, language
- [ ] opened the file once to check Arabic displays correctly

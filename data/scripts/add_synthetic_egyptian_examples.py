"""
add_synthetic_egyptian_examples.py

Adds ~50 synthetic Egyptian-brand-impersonation phishing examples, paired with
genuine-style legitimate examples from the same brands, to data/processed/dataset.csv.

These are grounded in real, currently-documented Egyptian fraud patterns reported
by the Central Bank of Egypt, Vodafone Egypt, Instapay, and Egyptian news outlets
(see docs/synthetic_examples_grounding.md for the source list) -- not generic
imagined phishing content. Per CLAUDE.md's scope guardrail, these are produced as
labeled detection/training data, not standalone reusable phishing templates.

Usage:
    python add_synthetic_egyptian_examples.py
"""

import csv
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language",
    "source", "brand_impersonated", "date_collected", "split",
]

# Each entry: (brand, content_type, language, label, text)
# Grounded patterns used:
#   - Fake prize/reward message (documented Bank Misr warning)
#   - Fake "sent by mistake, please return" message (documented Vodafone Cash scam)
#   - Fake account-update / new-service-activation request (documented Instapay scam)
#   - Fake pending-shipment link (documented CBE nationwide pattern)
#   - Fake OTP/PIN request via call or SMS (documented across all sources)
#   - Fake card-blocked message (pattern seen in the GitHub Arabic spam dataset)
EXAMPLES = [
    # --- Banque Misr ---
    ("Banque Misr", "text", "ar", "phishing",
     "تهنئة!! حسابك في بنك مصر مؤهل لجائزة نقدية 25000 جنيه ضمن حملة العملاء المميزين. لاستلام الجائزة يرجى تأكيد بياناتك عبر الرابط خلال 24 ساعة: bnkmisr-rewards.com"),
    ("Banque Misr", "text", "ar", "legitimate",
     "بنك مصر: تم خصم مبلغ 450.00 جنيه من حسابكم المنتهي بـ 4471 في تمام 10:32ص. للاستفسار اتصل بالخط الساخن 19888. لا تشارك بياناتك مع أي شخص."),

    # --- Vodafone Cash ---
    ("Vodafone Egypt", "text", "ar", "phishing",
     "تم تحويل مبلغ 1500 جنيه لمحفظتك بالخطأ من فودافون كاش. برجاء إعادة المبلغ فورا عن طريق الاتصال بالرقم المرفق وإدخال كود التحقق اللي هيوصلك."),
    ("Vodafone Egypt", "text", "ar", "phishing",
     "عميلنا العزيز، تطبيق أنا فودافون يحتاج تفعيل بياناتك خلال 48 ساعة وإلا سيتم إيقاف الخط. اضغط هنا: vf-egypt-verify.net"),
    ("Vodafone Egypt", "text", "ar", "legitimate",
     "فودافون: رصيدك الحالي 87.50 جنيه. لشحن رصيدك أو الاستعلام عن باقاتك استخدم تطبيق أنا فودافون الرسمي فقط. نحن لا نطلب كود OTP أبدا عبر الهاتف."),

    # --- Orange Egypt ---
    ("Orange Egypt", "text", "ar", "phishing",
     "أورانج: فاتورتك متأخرة السداد وسيتم قطع الخط خلال 24 ساعة. سدد الآن لتجنب الإيقاف: orange-eg-pay.com/bill"),
    ("Orange Egypt", "text", "mixed", "phishing",
     "Orange Egypt: Your account requires urgent verification. Click here to avoid suspension: orange-egypt-secure.net يرجى التحديث فورا"),
    ("Orange Egypt", "text", "ar", "legitimate",
     "أورانج مصر: فاتورتك لشهر يوليو جاهزة بقيمة 210 جنيه. يمكنك السداد عبر تطبيق My Orange أو أقرب فرع. تاريخ الاستحقاق 25/7."),

    # --- Etisalat Misr / e& ---
    ("Etisalat Misr (e&)", "email", "ar", "phishing",
     "عزيزي العميل، اشتراكك في خدمة e& الإضافية سيتجدد تلقائيا بمبلغ 99 جنيه. لإلغاء الاشتراك اضغط على الرابط وأدخل بيانات بطاقتك للتأكيد: eand-services-eg.com/cancel"),
    ("Etisalat Misr (e&)", "email", "ar", "legitimate",
     "عزيزي العميل، يسر e& مصر إعلامكم بإطلاق باقات جديدة للإنترنت المنزلي بأسعار مميزة. لمزيد من التفاصيل يرجى زيارة الموقع الرسمي أو أقرب فرع لنا."),

    # --- Telecom Egypt (WE) ---
    ("Telecom Egypt (WE)", "text", "ar", "phishing",
     "الدعم الفني لشركة WE: تم رصد مشكلة في خط الانترنت الخاص بك. يرجى الاتصال بالرقم فورا وتزويد فريق الدعم برمز التحقق لإصلاح المشكلة عن بعد."),
    ("Telecom Egypt (WE)", "text", "ar", "legitimate",
     "WE: نود إعلامكم بأعمال صيانة مجدولة على الشبكة في منطقتكم يوم الخميس من 2 إلى 4 صباحا. نعتذر عن أي إزعاج."),

    # --- Instapay ---
    ("Instapay", "text", "ar", "phishing",
     "إنستاباي: حسابك يحتاج تفعيل خدمة التحويل الفوري الجديدة. لإتمام التفعيل يرجى الدخول على الرابط وتسجيل بيانات حسابك البنكي: instapay-activate.com"),
    ("Instapay", "text", "ar", "phishing",
     "تنبيه من إنستاباي: تم رصد محاولة دخول غير معروفة على حسابك. لتأمين حسابك أدخل الرقم السري ورمز OTP على الرابط التالي فورا: instapay-secure-verify.net"),
    ("Instapay", "email", "ar", "legitimate",
     "تحذير أمني من إنستاباي: نود تذكيركم بأننا لا نطلب منكم أبدا مشاركة الرقم السري أو رمز التحقق OTP عبر الهاتف أو الرسائل النصية. أي طلب من هذا النوع يعتبر محاولة احتيال."),

    # --- Fawry ---
    ("Fawry", "text", "ar", "phishing",
     "فوري: فشلت عملية الدفع الأخيرة الخاصة بك. لتحديث بيانات البطاقة وتجنب إيقاف الخدمة اضغط هنا: fawry-update-card.com"),
    ("Fawry", "text", "ar", "legitimate",
     "فوري: تمت عملية الدفع بنجاح بمبلغ 350 جنيه لصالح فاتورة الكهرباء. رقم العملية: 88213456. شكرا لاستخدامكم فوري."),

    # --- CIB ---
    ("Commercial International Bank (CIB)", "email", "ar", "phishing",
     "عميلنا العزيز، تم اكتشاف نشاط غير معتاد على حسابك في بنك CIB. يرجى تسجيل الدخول فورا لتأكيد هويتك وتجنب تجميد الحساب: cib-eg-secure.com/login"),
    ("Commercial International Bank (CIB)", "email", "ar", "legitimate",
     "عميلنا العزيز، كشف حسابك الشهري في بنك CIB أصبح متاحا الآن عبر تطبيق CIB Smart Wallet أو الإنترنت البنكي. للاستفسار يرجى زيارة أقرب فرع."),

    # --- National Bank of Egypt (NBE) ---
    ("National Bank of Egypt (NBE)", "text", "ar", "phishing",
     "البنك الأهلي المصري: حسابك حصل على نقاط مكافآت بقيمة 500 جنيه. لاستلامها قم بتسجيل بيانات بطاقتك على: nbe-rewards-eg.com"),
    ("National Bank of Egypt (NBE)", "text", "ar", "legitimate",
     "الأهلي: كود التحقق الخاص بك هو 483920. لا تشارك هذا الكود مع أي شخص، حتى لو ادعى أنه موظف بالبنك."),

    # --- QNB Al Ahli ---
    ("QNB Al Ahli", "email", "ar", "phishing",
     "تنبيه أمني هام من QNB الأهلي: حسابك سيتم تعليقه خلال 24 ساعة بسبب معلومات ناقصة. أكمل بياناتك الآن لتفادي التعليق: qnbalahli-verify.com"),
    ("QNB Al Ahli", "email", "ar", "legitimate",
     "عميلنا العزيز، يسر QNB الأهلي دعوتكم لتجربة خدمة الإنترنت البنكي الجديدة عبر موقعنا الرسمي qnb.com.eg. لأي استفسار تواصلوا مع خدمة العملاء."),

    # --- Arab African International Bank (AAIB) ---
    ("Arab African International Bank", "text", "ar", "phishing",
     "عزيز العميل تم حظر بطاقتك الصراف الآلي الخاصة ببنك AAIB. لأنك لا تملك التحديث بعد. اتصل بهذا الرقم على الفور للتحديث."),
    ("Arab African International Bank", "text", "ar", "legitimate",
     "AAIB: تم إيداع راتبكم الشهري بنجاح في حسابكم. الرصيد الحالي متاح عبر تطبيق AAIB Mobile Banking."),

    # --- Jumia Egypt ---
    ("Jumia Egypt", "email", "ar", "phishing",
     "جوميا: طلبك رقم #4471 واجه مشكلة في الشحن. لإعادة جدولة التوصيل يرجى تأكيد عنوانك وبيانات الدفع عبر الرابط: jumia-eg-delivery.com"),
    ("Jumia Egypt", "email", "ar", "legitimate",
     "جوميا: تم شحن طلبك رقم #4471 وسيصلك خلال 2-3 أيام عمل. يمكنك تتبع الشحنة من خلال تطبيق جوميا أو موقعنا الرسمي."),

    # --- Noon Egypt ---
    ("Noon Egypt", "text", "ar", "phishing",
     "noon: طردك وصل الجمرك ومطلوب رسوم إضافية 85 جنيه لاستلامه. ادفع الآن عبر الرابط لتجنب إرجاع الشحنة: noon-eg-customs.com"),
    ("Noon Egypt", "text", "ar", "legitimate",
     "noon: طلبك في الطريق إليك! المندوب سيصل خلال ساعتين تقريبا. الدفع عند الاستلام كما طلبت."),

    # --- Amazon Egypt ---
    ("Amazon Egypt", "email", "en", "phishing",
     "Your Amazon.eg account has been locked due to unusual sign-in activity. Verify your identity now to restore access: amazon-eg-account-verify.com"),
    ("Amazon Egypt", "email", "en", "legitimate",
     "Your Amazon.eg order #703-2214 has shipped and is expected to arrive within 3-5 business days. Track your package anytime from Your Orders."),

    # --- Talabat ---
    ("Talabat Egypt", "text", "ar", "phishing",
     "طلبات: تم خصم مبلغ الطلب مرتين بالخطأ من محفظتك. لاسترداد المبلغ الزائد يرجى إدخال بيانات بطاقتك على الرابط: talabat-refund-eg.com"),
    ("Talabat Egypt", "text", "ar", "legitimate",
     "طلبات: تم تأكيد طلبك من مطعم كشري التحرير. الوقت المتوقع للتوصيل 35 دقيقة. يمكنك متابعة الطلب من التطبيق."),

    # --- Breadfast ---
    ("Breadfast", "text", "ar", "phishing",
     "Breadfast: تم تجديد اشتراكك الشهري بمبلغ 199 جنيه بدون موافقتك. لإلغاء الاشتراك واسترداد المبلغ اضغط هنا فورا: breadfast-billing.net"),
    ("Breadfast", "text", "ar", "legitimate",
     "Breadfast: طلبك اليومي في الطريق! هيوصلك خلال 20 دقيقة. تقدر تتابع أوردرك من التطبيق."),

    # --- EgyptAir ---
    ("EgyptAir", "email", "ar", "phishing",
     "مصر للطيران: تم إلغاء حجزك رقم MS-99213 بسبب مشكلة في الدفع. لإعادة تأكيد الحجز يرجى تحديث بيانات بطاقتك خلال 12 ساعة: egyptair-booking-confirm.com"),
    ("EgyptAir", "email", "ar", "legitimate",
     "مصر للطيران: تم تأكيد حجزكم على الرحلة MS-777 من القاهرة إلى جدة بتاريخ 15 أغسطس. نتمنى لكم رحلة سعيدة."),

    # --- Egypt e-Government portal ---
    ("Egypt e-Government portal", "text", "ar", "phishing",
     "منصة مصر الرقمية: تم اعتمادك للحصول على دعم مالي من الحكومة بقيمة 3000 جنيه. لصرف المبلغ يرجى تسجيل رقم حسابك البنكي: digital-gov-eg-support.com"),
    ("Egypt e-Government portal", "text", "ar", "legitimate",
     "منصة مصر الرقمية: تم استلام طلب تجديد بطاقة الرقم القومي الخاصة بكم بنجاح. يمكنكم استلامها من مكتب السجل المدني بعد 7 أيام عمل."),

    # --- Mixed brand-agnostic patterns (common social-engineering angles, CBE-documented) ---
    ("Multiple banks (generic)", "text", "ar", "phishing",
     "تنبيه من إدارة مكافحة الاحتيال: تم رصد محاولة سحب غير مصرح بها من حسابك. للتأكد من هويتك اتصل فورا على الرقم المرفق وقدم كود التحقق."),
    ("Multiple banks (generic)", "text", "ar", "phishing",
     "عزيزي العميل، بياناتك البنكية بحاجة لتحديث فوري بسبب تغيير في أنظمة الأمان. يرجى الاتصال بخدمة العملاء على الرقم المرسل في أقرب وقت."),
    ("Multiple banks (generic)", "text", "ar", "legitimate",
     "تذكير أمني: بنكك لن يطلب منك أبدا رقمك السري أو رمز OTP أو رقم بطاقتك كاملا عبر الهاتف أو الرسائل النصية. أي طلب كهذا هو محاولة احتيال، أبلغ البنك فورا."),
    ("Multiple banks (generic)", "text", "en", "phishing",
     "URGENT: Your bank account has been flagged for suspicious activity. Verify your identity immediately or your account will be suspended within 24 hours: secure-bank-verify-eg.com"),
    ("Multiple banks (generic)", "email", "mixed", "phishing",
     "Dear Customer, نود إعلامكم أن حسابكم البنكي معرض للإغلاق بسبب معلومات ناقصة. Please update your information immediately via the secure link below to avoid account suspension: bank-update-secure.net"),

    # --- WhatsApp-style (CBE-documented pattern: hacked contact requesting urgent transfer) ---
    ("WhatsApp impersonation (generic)", "text", "ar", "phishing",
     "أنا محتاج مساعدتك ضروري جدا، ممكن تحول لي 2000 جنيه دلوقتي على انستاباي وأنا هرجعهملك بكرة الصبح؟ الموضوع مستعجل ومش قادر أكلمك دلوقتي."),
    ("WhatsApp impersonation (generic)", "text", "ar", "legitimate",
     "لو حد بعتلك من رقم صاحبك أو قريبك بيطلب فلوس بشكل مستعجل من غير مكالمة صوتية، اتصل بيه مباشرة اتأكد الأول قبل ما تحول أي مبلغ."),

    # --- Additional difficulty-tiered phishing examples (harder to detect) ---
    ("Commercial International Bank (CIB)", "email", "ar", "phishing",
     "السيد/ مدير الحساب المحترم، نفيدكم بأنه تم تحديث سياسة الأمان الخاصة ببنك CIB اعتبارا من الشهر القادم. يرجى مراجعة المستند المرفق وتأكيد بيانات التوقيع الإلكتروني عبر الرابط أدناه قبل نهاية الأسبوع لضمان استمرارية الخدمة دون انقطاع: cib-policy-update.com/confirm"),
    ("Vodafone Egypt", "text", "ar", "phishing",
     "شكرا لتفعيل خدمة نقاط فودافون بلس. تم إضافة 1200 نقطة لحسابك. لاستبدال النقاط بخصومات فورية على فاتورتك، فعّل الخدمة من هنا خلال 3 أيام: vf-points-eg.com"),
    ("Instapay", "text", "en", "phishing",
     "Instapay Alert: A new device has been linked to your account. If this wasn't you, secure your account now: instapay-device-alert.com"),

    # --- More legitimate counterexamples for balance ---
    ("Orange Egypt", "email", "ar", "legitimate",
     "عميلنا العزيز، شكرا لتواصلكم مع خدمة عملاء أورانج. تم فتح تذكرة دعم فني برقم #55219 وسيتواصل معكم أحد ممثلينا خلال 24 ساعة."),
    ("Fawry", "email", "ar", "legitimate",
     "فوري: إيصال دفع إلكتروني - تم سداد فاتورة المياه بقيمة 120 جنيه بنجاح بتاريخ 14/7/2026. احتفظ بهذا الإيصال كمرجع."),
    ("National Bank of Egypt (NBE)", "email", "ar", "legitimate",
     "الأهلي المصري: تنويه توعوي - احذروا الرسائل التي تدعي فوزكم بجوائز أو تطلب تحديث بياناتكم عبر روابط. تواصلوا معنا فقط عبر الخط الساخن الرسمي 19623."),
    ("Talabat Egypt", "email", "ar", "legitimate",
     "طلبات: فاتورتك الإلكترونية - طلب رقم #98231 من سوبر ماركت الفا بقيمة 245 جنيه، تم الدفع كاش عند الاستلام. شكرا لاستخدامك طلبات."),
    ("EgyptAir", "text", "ar", "legitimate",
     "مصر للطيران: تذكير - رحلتك MS-777 غدا الساعة 6:30 صباحا. يرجى الحضور لمطار القاهرة قبل الموعد بـ 3 ساعات."),
]


def main() -> None:
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    existing_syn_count = sum(1 for row in existing_rows if row["id"].startswith("SYN-"))
    next_id_num = existing_syn_count + 1

    new_rows = []
    for i, (brand, content_type, language, label, text) in enumerate(EXAMPLES, start=next_id_num):
        new_rows.append({
            "id": f"SYN-{i:06d}",
            "text": text,
            "content_type": content_type,
            "label": label,
            "language": language,
            "source": "synthetic_egyptian_brands",
            "brand_impersonated": brand if label == "phishing" else "",
            "date_collected": date.today().isoformat(),
            "split": "",
        })

    all_rows = existing_rows + new_rows
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    phishing_count = sum(1 for r in new_rows if r["label"] == "phishing")
    legit_count = sum(1 for r in new_rows if r["label"] == "legitimate")

    print(f"Added {len(new_rows)} synthetic rows ({phishing_count} phishing, {legit_count} legitimate)")
    print(f"dataset.csv now has {len(all_rows):,} total rows")


if __name__ == "__main__":
    main()

# run_acceptance_test.ps1
#
# ETIS acceptance pass (Phase 9 checklist item): runs a fixed set of
# known inputs against the LIVE backend and records the actual
# verdict/score for each, so the result is a real, reproducible run --
# not eyeballed one-off testing.
#
# Prerequisite: backend must be running first, in a separate terminal:
#   cd backend
#   python -m uvicorn app.main:app --reload
#
# Usage (from anywhere, adjust path if needed):
#   powershell -File run_acceptance_test.ps1
#
# Output: prints a summary table to the console AND saves full raw
# JSON responses to acceptance_test_report.json in the current folder,
# so you have both a quick-read summary and the full engine breakdown
# for each case if you want to dig into one later.

$baseUrl = "http://localhost:8000"

$cases = @(
    @{ id = 1;  desc = "Known phishing URL (exact blacklist match)";       url = "https://allegrolokalnie.pl-5878761824523.lol/oferta/l/-130425072"; text = $null; expected = "HIGH" }
    @{ id = 2;  desc = "Known legitimate URL (verified Egyptian brand)";   url = "https://www.cibeg.com/personal-banking"; text = $null; expected = "LOW" }
    @{ id = 3;  desc = "Domain-embedding typosquat (not blacklisted)";     url = "cibeg.com.verify-account-now.net"; text = $null; expected = "MEDIUM" }
    @{ id = 4;  desc = "Homoglyph domain (Cyrillic e)";                    url = ("cib" + [char]0x0435 + "g.com"); text = $null; expected = "MEDIUM/HIGH" }
    @{ id = 5;  desc = "Arabic phishing text";                             url = $null; text = "بطاقتك محظورة يرجى ادخال كود التحقق فورا"; expected = "HIGH" }
    @{ id = 6;  desc = "Arabic legitimate text";                          url = $null; text = "تم إيداع راتبكم الشهري بنجاح في حسابكم"; expected = "LOW" }
    @{ id = 7;  desc = "English phishing text (lottery/advance-fee)";      url = $null; text = "Congratulations! You have been selected to win `$5000. Reply with your bank account number to claim."; expected = "HIGH" }
    @{ id = 8;  desc = "English legitimate text";                         url = $null; text = "Thanks for your order, it will arrive within 3-5 business days."; expected = "LOW" }
    @{ id = 9;  desc = "Mixed-language phishing text";                    url = $null; text = "مبروك! You have won a prize, احصل عليه الآن via this link"; expected = "MEDIUM/HIGH" }
    @{ id = 10; desc = "IP-address-host URL (no blacklist match)";         url = "http://192.168.1.50/login"; text = $null; expected = "MEDIUM" }
)

$results = @()

Write-Host "`nETIS Acceptance Pass -- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Cyan

foreach ($case in $cases) {
    $body = @{ url = $case.url; text = $case.text } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/analyze" -Method Post -Body $body -ContentType "application/json; charset=utf-8"
        $actualVerdict = $response.verdict
        $actualScore = $response.final_score
        $status = "OK"
    } catch {
        $actualVerdict = "ERROR"
        $actualScore = $null
        $status = $_.Exception.Message
        $response = $null
    }

    $results += [PSCustomObject]@{
        Id       = $case.id
        Desc     = $case.desc
        Expected = $case.expected
        Actual   = $actualVerdict
        Score    = $actualScore
        Status   = $status
        Raw      = $response
    }

    $color = if ($actualVerdict -eq "ERROR") { "Red" } else { "White" }
    Write-Host ("[{0,2}] {1,-45} expected={2,-12} actual={3,-8} score={4}" -f $case.id, $case.desc, $case.expected, $actualVerdict, $actualScore) -ForegroundColor $color
}

# Save full raw responses for later inspection
$results | Select-Object Id, Desc, Expected, Actual, Score, Status, Raw | ConvertTo-Json -Depth 10 | Out-File -FilePath "acceptance_test_report.json" -Encoding utf8

Write-Host "`nFull report (including per-engine breakdowns) saved to acceptance_test_report.json`n" -ForegroundColor Cyan

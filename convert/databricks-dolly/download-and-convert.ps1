# Convert databricks-dolly-15k.txt from JSONL format to Blip tuning format
# Usage: .\convert-to-blip.ps1 -InputFile databricks-dolly-15k.txt -OutputFile dolly-tuning.txt

param(
    [string]$InputFile = "",
    [string]$SourceUrl = "https://cas-bridge.xethub.hf.co/xet-bridge-us/64358e2179c45fcf1ada09f4/63c4dabe683d7254493568d2d3995c0e51abc8528ef3b4936497c538cb501e93?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260504%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260504T131654Z&X-Amz-Expires=3600&X-Amz-Signature=f5515299ca5f1dfb9864b41cb1116ae7d9bbdf3a215bdf761e83b7e052752556&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27databricks-dolly-15k.jsonl%3B+filename%3D%22databricks-dolly-15k.jsonl%22%3B&x-amz-checksum-mode=ENABLED&x-id=GetObject&Expires=1777904214&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NzkwNDIxNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82NDM1OGUyMTc5YzQ1ZmNmMWFkYTA5ZjQvNjNjNGRhYmU2ODNkNzI1NDQ5MzU2OGQyZDM5OTVjMGU1MWFiYzg1MjhlZjNiNDkzNjQ5N2M1MzhjYjUwMWU5MyoifV19&Signature=bFx8jW7Xjb27KLodGDdwzCldAUgLbdPSUPt5lAjCVRO-mOrWKx-fB1AvInTlRTt8F%7Ezg4x0mbTSyjqIqkKZsGDQbN7vQP4LlvkO2iYbiYKcML833Vb%7ErEgUvLX%7EKecEJie5a-HcHOyppgI2ooGNAiM-MGs7Flm6X1L-FIk0nse-2v2pMNlUmn%7EB8lzlQ5IoJW7Ac2krKf7dOeQkD9mkPr4bDMgREIVO4vb5z-OGUKMWBsFULJN7HZh-B5IZA3u7iRzHsfwRNFxpat2rwaWdfbuo%7EaSe%7Eo2ch2UjW5--iByF8D1TEoiWjmePB9wXC12vNYqe4a1UUtQrgbN1znbliug__&Key-Pair-Id=K2L8F4GPSG1IFC",
    [string]$DownloadPath = "$env:TEMP\databricks-dolly-15k.txt",
    [string]$OutputFile = "..\..\training\tuning\dolly-tuning.txt"
)

# Download source file by default when InputFile is not provided
$downloadedInput = $false
if ([string]::IsNullOrWhiteSpace($InputFile)) {
    $InputFile = $DownloadPath
    $downloadDir = Split-Path -Parent $InputFile
    if (-not [string]::IsNullOrWhiteSpace($downloadDir) -and -not (Test-Path $downloadDir)) {
        New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    }

    Write-Host "Downloading dataset from source URL..."
    try {
        Invoke-WebRequest -Uri $SourceUrl -OutFile $InputFile
        $downloadedInput = $true
    }
    catch {
        Write-Error "Failed to download source dataset: $_"
        exit 1
    }
}
elseif (-not (Test-Path $InputFile)) {
    Write-Error "Input file not found: $InputFile"
    exit 1
}

Write-Host "Converting $InputFile to Blip tuning format..."

function Convert-ToSingleLineBlipText {
    param(
        [AllowNull()]
        [string]$Text
    )

    if ($null -eq $Text) {
        return ""
    }

    return ($Text -replace "`r`n|`n|`r", "\\n")
}

$output = @()
$output += "# Source: Databricks Dolly 15K"
$output += "# Source file: $InputFile"
$output += "# Source URL: https://cas-bridge.xethub.hf.co/xet-bridge-us/64358e2179c45fcf1ada09f4/63c4dabe683d7254493568d2d3995c0e51abc8528ef3b4936497c538cb501e93?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260504%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260504T131654Z&X-Amz-Expires=3600&X-Amz-Signature=f5515299ca5f1dfb9864b41cb1116ae7d9bbdf3a215bdf761e83b7e052752556&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27databricks-dolly-15k.jsonl%3B+filename%3D%22databricks-dolly-15k.jsonl%22%3B&x-amz-checksum-mode=ENABLED&x-id=GetObject&Expires=1777904214&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NzkwNDIxNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82NDM1OGUyMTc5YzQ1ZmNmMWFkYTA5ZjQvNjNjNGRhYmU2ODNkNzI1NDQ5MzU2OGQyZDM5OTVjMGU1MWFiYzg1MjhlZjNiNDkzNjQ5N2M1MzhjYjUwMWU5MyoifV19&Signature=bFx8jW7Xjb27KLodGDdwzCldAUgLbdPSUPt5lAjCVRO-mOrWKx-fB1AvInTlRTt8F%7Ezg4x0mbTSyjqIqkKZsGDQbN7vQP4LlvkO2iYbiYKcML833Vb%7ErEgUvLX%7EKecEJie5a-HcHOyppgI2ooGNAiM-MGs7Flm6X1L-FIk0nse-2v2pMNlUmn%7EB8lzlQ5IoJW7Ac2krKf7dOeQkD9mkPr4bDMgREIVO4vb5z-OGUKMWBsFULJN7HZh-B5IZA3u7iRzHsfwRNFxpat2rwaWdfbuo%7EaSe%7Eo2ch2UjW5--iByF8D1TEoiWjmePB9wXC12vNYqe4a1UUtQrgbN1znbliug__&Key-Pair-Id=K2L8F4GPSG1IFC"
$output += ""
$lineCount = 0
$skipCount = 0

# Read and parse JSONL file
Get-Content $InputFile | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) {
        return
    }
    
    try {
        $json = $_ | ConvertFrom-Json
        $lineCount++
        
        # Extract instruction and response
        $instruction = Convert-ToSingleLineBlipText $json.instruction
        $response = Convert-ToSingleLineBlipText $json.response
        
        # Skip if either is missing or too short
        if ([string]::IsNullOrWhiteSpace($instruction) -or [string]::IsNullOrWhiteSpace($response)) {
            $skipCount++
            return
        }
        
        # Add user/ai pair with blank line separator
        $output += "user:$instruction"
        $output += "ai:$response"
        $output += ""
    }
    catch {
        $skipCount++
        Write-Warning "Skipped invalid JSON on line $lineCount`:"
    }
}

# Write output file
$outputDir = Split-Path -Parent $OutputFile
if (-not [string]::IsNullOrWhiteSpace($outputDir) -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$output | Out-File -FilePath $OutputFile -Encoding UTF8

if ($downloadedInput -and (Test-Path $InputFile)) {
    Remove-Item $InputFile -Force
}

Write-Host "✓ Conversion complete!"
Write-Host "  Processed: $lineCount lines"
Write-Host "  Skipped: $skipCount lines"
Write-Host "  Output: $OutputFile"

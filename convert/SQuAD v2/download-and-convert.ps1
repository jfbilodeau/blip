# Convert SQuAD v2.0 JSON to Blip tuning format
# Usage: .\convert-to-blip.ps1 -InputFile train-v2.0.json

param(
    [string]$InputFile = "",
    [string]$SourceUrl = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    [string]$DownloadPath = "$env:TEMP\train-v2.0.json",
    [string]$OutputFile = "..\..\training\tuning\squad-tuning.txt"
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
$output += "# Source: SQuAD v2.0"
$output += "# Source file: $InputFile"
$output += "# Source URL: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
$output += ""
$questionCount = 0
$skipCount = 0

try {
    # Read and parse JSON file
    $json = Get-Content $InputFile -Raw | ConvertFrom-Json
    
    # Iterate through articles
    foreach ($article in $json.data) {
        # Iterate through paragraphs
        foreach ($paragraph in $article.paragraphs) {
            # Iterate through questions
            foreach ($qa in $paragraph.qas) {
                $questionCount++
                
                # Skip if no answers or is_impossible
                if ($qa.is_impossible -eq $true -or $qa.answers.Count -eq 0) {
                    $skipCount++
                    continue
                }
                
                $question = Convert-ToSingleLineBlipText $qa.question
                $answer = Convert-ToSingleLineBlipText $qa.answers[0].text
                
                # Skip if either is empty
                if ([string]::IsNullOrWhiteSpace($question) -or [string]::IsNullOrWhiteSpace($answer)) {
                    $skipCount++
                    continue
                }
                
                # Add user/ai pair with blank line separator
                $output += "user:$question"
                $output += "ai:$answer"
                $output += ""
            }
        }
    }
}
catch {
    Write-Error "Failed to parse JSON file: $_"
    exit 1
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
Write-Host "  Questions processed: $questionCount"
Write-Host "  Skipped: $skipCount"
Write-Host "  Output: $OutputFile"

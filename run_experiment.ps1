# ============================================================
#  Experiment runner — N seeded runs per instance, 4 parallel islands
#  Usage:
#    Single instance:
#      powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instances 0
#    Multiple instances:
#      powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instances 0,1,2,3
# ============================================================

param(
    [int[]]$Instances  = @(0),
    [int]  $Islands    = 4,
    [int]  $Iterations = 200
)

$EXE    = "C:\Users\Demir\researchproject\MA-CETSP\build\Release\MA-CETSP.exe"
$OUTDIR = "C:\Users\Demir\researchproject\MA-CETSP\solutions\experiment_results"
$SEEDS  = @(1, 11, 21, 31, 41, 51, 61, 71, 81, 91)

New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

$totalRuns = $Instances.Count * $SEEDS.Count
$globalRun = 0

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host " Instances  : $($Instances -join ', ')" -ForegroundColor Yellow
Write-Host " Seeds      : $($SEEDS -join ', ')" -ForegroundColor Yellow
Write-Host " Islands    : $Islands" -ForegroundColor Yellow
Write-Host " Iterations : $Iterations" -ForegroundColor Yellow
Write-Host " Total runs : $totalRuns" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

# ---- helper: run one instance through all seeds ----
function Run-Instance {
    param([int]$Instance)

    $instDir     = "$OUTDIR\instance$Instance"
    New-Item -ItemType Directory -Force -Path $instDir | Out-Null
    $SummaryFile = "$instDir\summary.txt"

    $header = "============================================================`n"
    $header += " Instance $Instance  |  $Islands islands  |  $Iterations iterations`n"
    $header += " Seeds : $($SEEDS -join ', ')`n"
    $header += " Started : $(Get-Date)`n"
    $header += "============================================================`n`n"
    $header | Set-Content -Path $SummaryFile -Encoding utf8

    $allRuns = @()

    for ($r = 0; $r -lt $SEEDS.Count; $r++) {
        $seed   = $SEEDS[$r]
        $runNum = $r + 1
        $script:globalRun++
        $runLog = "$instDir\run${runNum}_seed${seed}.log"

        Write-Host "[Instance $Instance | Run $runNum/$($SEEDS.Count) | Overall $_globalRun/$totalRuns] seed=$seed ..." -ForegroundColor Cyan

        & $EXE -i $Instance -s $seed --islands $Islands -r $Iterations 2>&1 |
            Out-File -FilePath $runLog -Encoding utf8

        $lines = Get-Content $runLog

        # Parse per-island lines
        $islandResults = @()
        foreach ($line in $lines) {
            if ($line -match '^\s+Island\s+(\d+)\s+model=(\S+)\s+best=([\d.]+)\s+time=([\d.]+)s\s+rejected=(\d+)') {
                $islandResults += [PSCustomObject]@{
                    Island   = [int]$Matches[1]
                    Model    = $Matches[2]
                    Best     = [double]$Matches[3]
                    Time     = [double]$Matches[4]
                    Rejected = [int]$Matches[5]
                }
            }
        }

        # Parse global best and wall-clock
        $globalBest  = $null
        $globalModel = $null
        $globalTime  = $null
        $wallClock   = $null
        foreach ($line in $lines) {
            if ($line -match 'GLOBAL BEST:.*model=(\S+)\s+value=([\d.]+)\s+time=([\d.]+)s') {
                $globalModel = $Matches[1]
                $globalBest  = [double]$Matches[2]
                $globalTime  = [double]$Matches[3]
            }
            if ($line -match 'Wall-clock.*?:\s*([\d.]+)s') {
                $wallClock = [double]$Matches[1]
            }
        }

        $allRuns += [PSCustomObject]@{
            Run         = $runNum
            Seed        = $seed
            Islands     = $islandResults
            GlobalBest  = $globalBest
            GlobalModel = $globalModel
            WallClock   = $wallClock
        }

        # Write run block
        $block  = "------------------------------------------------------------`n"
        $block += "Run $runNum  |  seed=$seed`n"
        $block += "------------------------------------------------------------`n"
        foreach ($isl in $islandResults) {
            $block += "  Island $($isl.Island)  model=$($isl.Model.PadRight(10))"
            $block += "  best=$($isl.Best.ToString('F3').PadLeft(10))"
            $block += "  time=$($isl.Time.ToString('F3').PadLeft(8))s"
            $block += "  rejected=$($isl.Rejected)`n"
        }
        $block += "  --`n"
        $block += "  GLOBAL BEST  model=$globalModel  value=$($globalBest.ToString('F3'))  time=${globalTime}s  wall-clock=${wallClock}s`n`n"

        $block | Add-Content -Path $SummaryFile -Encoding utf8

        Write-Host "  -> best=$globalBest  model=$globalModel  wall-clock=${wallClock}s"
    }

    # ---- Aggregate statistics per model ----
    $statsBlock  = "`n============================================================`n"
    $statsBlock += " Aggregate Statistics (across $($SEEDS.Count) runs)`n"
    $statsBlock += "============================================================`n"

    $modelNames = $allRuns[0].Islands | Select-Object -ExpandProperty Model

    foreach ($model in $modelNames) {
        $bests = $allRuns | ForEach-Object {
            ($_.Islands | Where-Object { $_.Model -eq $model }).Best
        } | Where-Object { $_ -ne $null }

        $times = $allRuns | ForEach-Object {
            ($_.Islands | Where-Object { $_.Model -eq $model }).Time
        } | Where-Object { $_ -ne $null }

        $wins = ($allRuns | Where-Object { $_.GlobalModel -eq $model }).Count

        $meanBest = ($bests | Measure-Object -Average).Average
        $minBest  = ($bests | Measure-Object -Minimum).Minimum
        $maxBest  = ($bests | Measure-Object -Maximum).Maximum
        $stdBest  = [math]::Sqrt(($bests | ForEach-Object { ($_ - $meanBest) * ($_ - $meanBest) } | Measure-Object -Average).Average)
        $meanTime = ($times | Measure-Object -Average).Average

        $statsBlock += "`n  Model: $model`n"
        $statsBlock += "    Best   mean: $($meanBest.ToString('F3'))  std: $($stdBest.ToString('F3'))  min: $($minBest.ToString('F3'))  max: $($maxBest.ToString('F3'))`n"
        $statsBlock += "    Time   avg: $($meanTime.ToString('F3'))s`n"
        $statsBlock += "    Wins   $wins / $($SEEDS.Count)`n"
    }

    $statsBlock += "`n============================================================`n"
    $statsBlock | Add-Content -Path $SummaryFile -Encoding utf8

    Write-Host ""
    Write-Host "Instance $Instance done. Summary -> $SummaryFile" -ForegroundColor Green
    Write-Host ""
}

# ---- main loop over all instances ----
foreach ($inst in $Instances) {
    Run-Instance -Instance $inst
}

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host " All done. $totalRuns total runs completed." -ForegroundColor Yellow
Write-Host " Results in: $OUTDIR" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

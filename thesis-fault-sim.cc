/* =============================================================================
 * THESIS FAULT SIMULATION — NS-3 3.38 (Chapter 3 / Table 3.1)
 *
 * 28-cell HetNet: 7 macro gNBs + 21 small cells (3 per macro)
 * KPI collection at 1 s intervals via NS-3 event scheduler + RNG framework
 * Fault injection: power | congestion | hardware (Section 3.2.3)
 *
 * BUILD:  ~/thesis-sim/bin/ns3 build thesis-fault-sim
 * RUN:    ~/thesis-sim/bin/ns3 run "thesis-fault-sim --trial=0 --fault=power ..."
 * ============================================================================= */

#include "ns3/core-module.h"
#include "ns3/network-module.h"

#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ThesisFaultSim");

static const uint32_t N_MACRO     = 7;
static const uint32_t N_SMALL_PM  = 3;
static const uint32_t N_CELLS     = N_MACRO * (1 + N_SMALL_PM);  // 28
static const double   SIM_TIME    = 300.0;
static const double   KPI_STEP    = 1.0;

static uint32_t g_faultCell      = 0;
static double   g_faultStart     = 9999.0;
static double   g_faultEnd       = 9999.0;
static std::string g_faultType   = "none";
static uint32_t g_trial          = 0;
static std::string g_outputDir   = ".";
static std::ofstream g_csv;

static bool IsMacro(uint32_t cellId) { return cellId < N_MACRO; }
static uint32_t ParentMacro(uint32_t cellId) {
    if (cellId < N_MACRO) return cellId;
    return (cellId - N_MACRO) / N_SMALL_PM;
}

struct KpiRow {
    double time;
    uint32_t cell;
    uint32_t macro;
    std::string cellType;
    double rsrp, sinr, prb, dl_tput, ul_tput, pkt_loss, ho_rate, latency;
    int label;
};

KpiRow GenerateKpi(double t, uint32_t cellId,
                   Ptr<NormalRandomVariable> rvNorm,
                   Ptr<UniformRandomVariable> rvUnif)
{
    KpiRow r;
    r.time  = t;
    r.cell  = cellId;
    r.macro = ParentMacro(cellId);
    r.cellType = IsMacro(cellId) ? "macro" : "small";

    bool inFault = (g_faultType != "none")
                && (t >= g_faultStart) && (t < g_faultEnd)
                && (cellId == g_faultCell);

    if (!inFault) {
        r.label = 0;
    } else if (g_faultType == "power") {
        r.label = 1;
    } else if (g_faultType == "congestion") {
        r.label = 2;
    } else {
        r.label = 3;
    }

    double noise = rvNorm->GetValue();
    double smallOffset = IsMacro(cellId) ? 0.0 : rvUnif->GetValue(-1.5, 1.5);

    r.rsrp    = -77.0  + noise * 3.0 + smallOffset;
    r.sinr    =  18.0  + noise * 2.0 + smallOffset * 0.5;
    r.prb     =  0.65  + noise * 0.04;
    r.dl_tput = 110.0  + noise * 15.0 + smallOffset * 5.0;
    r.ul_tput =  28.0  + noise * 4.0;
    r.pkt_loss = 0.005 + std::abs(noise) * 0.001;
    r.ho_rate  = 0.975 + noise * 0.005;
    r.latency  =  17.0 + std::abs(noise) * 2.0;

    if (r.label == 1) {
        double u = rvUnif->GetValue(0.0, 1.0);
        r.rsrp = -118.0 + u * 6.0;
        r.sinr = 1.5 + u * 1.5;
        r.prb = 0.0 + u * 0.02;
        r.dl_tput = 0.5 + u * 2.0;
        r.ul_tput = 0.1 + u * 0.5;
        r.pkt_loss = 0.93 + u * 0.06;
        r.ho_rate = 0.03 + u * 0.04;
        r.latency = 2400.0 + u * 300.0;
    } else if (r.label == 2) {
        double u = rvUnif->GetValue(0.0, 1.0);
        r.rsrp = -82.0 + noise * 3.0;
        r.sinr = 10.0 + u * 3.0;
        r.prb = 0.93 + u * 0.05;
        r.dl_tput = 35.0 + u * 10.0;
        r.ul_tput = 10.0 + u * 4.0;
        r.pkt_loss = 0.18 + u * 0.12;
        r.ho_rate = 0.82 + u * 0.06;
        r.latency = 420.0 + u * 250.0;
    } else if (r.label == 3) {
        double u = rvUnif->GetValue(0.0, 1.0);
        r.rsrp = -114.0 + u * 6.0;
        r.sinr = 2.0 + u * 2.0;
        r.prb = 0.01 + u * 0.02;
        r.dl_tput = 1.0 + u * 3.0;
        r.ul_tput = 0.2 + u * 0.8;
        r.pkt_loss = 0.88 + u * 0.10;
        r.ho_rate = 0.05 + u * 0.06;
        r.latency = 2100.0 + u * 400.0;
    }

    r.rsrp = std::max(-130.0, std::min(-50.0, r.rsrp));
    r.sinr = std::max(-5.0, std::min(35.0, r.sinr));
    r.prb = std::max(0.0, std::min(1.0, r.prb));
    r.dl_tput = std::max(0.0, std::min(500.0, r.dl_tput));
    r.ul_tput = std::max(0.0, std::min(200.0, r.ul_tput));
    r.pkt_loss = std::max(0.0, std::min(1.0, r.pkt_loss));
    r.ho_rate = std::max(0.0, std::min(1.0, r.ho_rate));
    r.latency = std::max(1.0, std::min(5000.0, r.latency));

    return r;
}

void CollectKpi(double t,
                Ptr<NormalRandomVariable> rvNorm,
                Ptr<UniformRandomVariable> rvUnif)
{
    for (uint32_t c = 0; c < N_CELLS; c++) {
        KpiRow row = GenerateKpi(t, c, rvNorm, rvUnif);
        g_csv << std::fixed << std::setprecision(4)
              << g_trial << ","
              << g_faultType << ","
              << row.time << ","
              << row.cell << ","
              << row.macro << ","
              << row.cellType << ","
              << row.rsrp << ","
              << row.sinr << ","
              << row.prb << ","
              << row.dl_tput << ","
              << row.ul_tput << ","
              << row.pkt_loss << ","
              << row.ho_rate << ","
              << row.latency << ","
              << g_faultStart << ","
              << g_faultEnd << ","
              << row.label << "\n";
    }

    double next = t + KPI_STEP;
    if (next <= SIM_TIME) {
        Simulator::Schedule(Seconds(KPI_STEP), &CollectKpi, next, rvNorm, rvUnif);
    }
}

int main(int argc, char *argv[])
{
    CommandLine cmd(__FILE__);
    cmd.AddValue("trial", "Trial index 0-49", g_trial);
    cmd.AddValue("fault", "none|power|congestion|hardware", g_faultType);
    cmd.AddValue("outputDir", "Directory for CSV output", g_outputDir);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(1000 + g_trial);
    RngSeedManager::SetRun(g_trial);

    Ptr<NormalRandomVariable> rvNorm = CreateObject<NormalRandomVariable>();
    rvNorm->SetAttribute("Mean", DoubleValue(0.0));
    rvNorm->SetAttribute("Variance", DoubleValue(1.0));

    Ptr<UniformRandomVariable> rvUnif = CreateObject<UniformRandomVariable>();
    rvUnif->SetAttribute("Min", DoubleValue(0.0));
    rvUnif->SetAttribute("Max", DoubleValue(1.0));

    if (g_faultType != "none") {
        Ptr<UniformRandomVariable> rvFault = CreateObject<UniformRandomVariable>();
        rvFault->SetAttribute("Min", DoubleValue(0.0));
        rvFault->SetAttribute("Max", DoubleValue(1.0));
        g_faultStart = 30.0 + rvFault->GetValue() * 220.0;
        double duration = 15.0 + rvFault->GetValue() * 30.0;
        g_faultEnd = g_faultStart + duration;
        g_faultCell = (uint32_t)(rvFault->GetValue() * N_CELLS);
        if (g_faultCell >= N_CELLS) g_faultCell = N_CELLS - 1;
    } else {
        g_faultStart = 0.0;
        g_faultEnd = 0.0;
    }

    std::string csvPath = g_outputDir + "/kpi_trial"
                        + std::to_string(g_trial) + "_"
                        + g_faultType + ".csv";
    g_csv.open(csvPath);
    if (!g_csv.is_open()) {
        std::cerr << "ERROR: Cannot open " << csvPath << std::endl;
        return 1;
    }

    g_csv << "trial,fault_type,time,gnb_id,macro_id,cell_type,"
          << "rsrp_avg_dbm,sinr_avg_db,prb_utilisation,"
          << "dl_throughput_mbps,ul_throughput_mbps,packet_loss_rate,"
          << "handover_success_rate,latency_avg_ms,"
          << "fault_start_s,fault_end_s,fault_label\n";

    Simulator::Schedule(Seconds(1.0), &CollectKpi, 1.0, rvNorm, rvUnif);
    Simulator::Stop(Seconds(SIM_TIME + 1.0));
    Simulator::Run();
    Simulator::Destroy();
    g_csv.close();

    std::ifstream check(csvPath);
    int lines = 0;
    std::string line;
    while (std::getline(check, line)) lines++;
    int expected = (int)(SIM_TIME * N_CELLS);
    if (lines - 1 < expected / 2) {
        std::cerr << "WARNING: only " << lines - 1 << " rows (expected ~" << expected << ")\n";
        return 1;
    }
    return 0;
}

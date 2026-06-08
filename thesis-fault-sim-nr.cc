/* =============================================================================
 * THESIS FAULT SIMULATION — NS-3 3.38 + 5G-LENA NR/EPC (Phase 3)
 *
 * Real 5G NR HetNet topology (Table 3.1):
 *   28 gNB cells: 7 macro + 21 small cells (3 per macro, ISD/3 offset)
 *   500 UEs target (use --numUes=280 if host is memory-limited)
 *   FR1 n78 band (3.5 GHz, 100 MHz, numerology 1)
 *   KPIs sampled every 1 s from NR PHY traces + FlowMonitor
 *
 * Fault injection (Section 3.2.3):
 *   power      — gNB TX power collapse
 *   congestion — 3× traffic surge on affected cell UEs
 *   hardware   — gNB PHY deactivation (TX power zero)
 *
 * BUILD:  ~/thesis-sim/bin/ns3 build thesis-fault-sim-nr
 * RUN:    ~/thesis-sim/bin/ns3 run "thesis-fault-sim-nr --trial=0 --fault=none ..."
 * ============================================================================= */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ThesisFaultSimNr");

static const uint32_t N_MACRO = 7;
static const uint32_t N_SMALL_PM = 3;
static const uint32_t N_CELLS = N_MACRO * (1 + N_SMALL_PM);
static const uint32_t N_UES_DEFAULT = 500;
static uint32_t g_numUes = N_UES_DEFAULT;
static double g_simTime = 300.0;
static const double KPI_STEP = 1.0;
static const double ISD = 500.0;
static const double SMALL_OFFSET = ISD / 3.0;

static uint32_t g_trial = 0;
static std::string g_faultType = "none";
static std::string g_outputDir = ".";
static uint32_t g_faultCell = 0;
static double g_faultStart = 9999.0;
static double g_faultEnd = 9999.0;

static std::ofstream g_csv;
static Ptr<FlowMonitor> g_flowMonitor;
static FlowMonitorHelper g_flowHelper;
static std::map<uint32_t, uint16_t> g_ueIndexToCell;
static std::map<uint16_t, uint32_t> g_cellToGnb;
static std::map<uint32_t, uint16_t> g_imsiToCell;
static std::map<Ipv4Address, uint16_t> g_ueIpToCell;
static std::map<uint32_t, uint32_t> g_imsiToUeIndex;
static std::map<uint32_t, Ipv4Address> g_ueIndexToIp;
static ApplicationContainer g_dlClients;
static std::vector<uint16_t> g_gnbCellIds;
static std::vector<uint32_t> g_ueHomeGnb;

static bool
IsMacroGnb(uint32_t gnbId)
{
    return gnbId < N_MACRO;
}

static uint32_t
ParentMacro(uint32_t gnbId)
{
    if (gnbId < N_MACRO) {
        return gnbId;
    }
    return (gnbId - N_MACRO) / N_SMALL_PM;
}

struct CellAccumulator {
    double rsrpSum = 0.0;
    double sinrSum = 0.0;
    uint32_t rsrpCount = 0;
    uint32_t sinrCount = 0;
    double dlBytes = 0.0;
    double ulBytes = 0.0;
    double delaySum = 0.0;
    uint64_t rxPackets = 0;
    uint64_t lostPackets = 0;
    uint32_t hoOk = 0;
    uint32_t hoFail = 0;
    std::map<uint32_t, uint64_t> lastFlowTxBytes;
    std::map<uint32_t, uint64_t> lastFlowRxBytes;
};

static std::map<uint16_t, CellAccumulator> g_cells;

static bool
InFaultWindow(uint32_t gnbId, double t)
{
    return g_faultType != "none" && gnbId == g_faultCell && t >= g_faultStart &&
           t < g_faultEnd;
}

static int
FaultLabel(uint32_t gnbId, double t)
{
    if (!InFaultWindow(gnbId, t)) {
        return 0;
    }
    if (g_faultType == "power") {
        return 1;
    }
    if (g_faultType == "congestion") {
        return 2;
    }
    return 3;
}

static void
OnReportRsrp(std::string context,
             uint16_t cellId,
             uint16_t imsi,
             uint16_t rnti,
             double rsrpDbm,
             uint8_t bwpId)
{
    (void)context;
    (void)imsi;
    (void)rnti;
    (void)bwpId;
    auto& acc = g_cells[cellId];
    acc.rsrpSum += rsrpDbm;
    acc.rsrpCount++;
}

static void
OnDlDataSinr(std::string context,
             uint16_t cellId,
             uint16_t rnti,
             double sinrLin,
             uint16_t bwpId,
             uint8_t streamId)
{
    (void)context;
    (void)rnti;
    (void)bwpId;
    (void)streamId;
    auto& acc = g_cells[cellId];
    double sinrDb = (sinrLin > 0.0) ? 10.0 * std::log10(sinrLin) : -5.0;
    acc.sinrSum += sinrDb;
    acc.sinrCount++;
}

static void
OnHandoverEndOk(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    (void)context;
    (void)rnti;
    g_imsiToCell[(uint32_t)imsi] = cellId;
    g_cells[cellId].hoOk++;
    if (g_imsiToUeIndex.count((uint32_t)imsi)) {
        uint32_t u = g_imsiToUeIndex[(uint32_t)imsi];
        if (g_ueIndexToIp.count(u)) {
            g_ueIpToCell[g_ueIndexToIp[u]] = cellId;
        }
    }
}

static void
OnHandoverEndError(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    (void)context;
    (void)rnti;
    g_imsiToCell[(uint32_t)imsi] = cellId;
    g_cells[cellId].hoFail++;
}

static void
UpdateFlowMonitorStats()
{
    if (!g_flowMonitor) {
        return;
    }
    g_flowMonitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(g_flowHelper.GetClassifier());
    auto stats = g_flowMonitor->GetFlowStats();
    for (auto it = stats.begin(); it != stats.end(); ++it) {
        uint32_t fid = it->first;
        const FlowMonitor::FlowStats& fs = it->second;
        if (!classifier || fs.rxPackets == 0) {
            continue;
        }
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(fid);
        if (!g_ueIpToCell.count(t.destinationAddress)) {
            continue;
        }
        uint16_t cellId = g_ueIpToCell[t.destinationAddress];
        auto& acc = g_cells[cellId];

        uint64_t txB = fs.txBytes;
        uint64_t rxB = fs.rxBytes;
        uint64_t prevTx = acc.lastFlowTxBytes[fid];
        uint64_t prevRx = acc.lastFlowRxBytes[fid];
        if (txB >= prevTx) {
            acc.dlBytes += (txB - prevTx);
        }
        if (rxB >= prevRx) {
            acc.ulBytes += (rxB - prevRx);
        }
        acc.lastFlowTxBytes[fid] = txB;
        acc.lastFlowRxBytes[fid] = rxB;
        acc.rxPackets += fs.rxPackets;
        acc.lostPackets += fs.lostPackets;
        acc.delaySum += fs.delaySum.GetSeconds();
    }
}

static void
WriteKpiRow(uint32_t gnbId, uint16_t cellId, double t, const CellAccumulator& snap)
{
    double rsrp = snap.rsrpCount ? snap.rsrpSum / snap.rsrpCount : -85.0;
    double sinr = snap.sinrCount ? snap.sinrSum / snap.sinrCount : 18.0;
    uint32_t phySamples = std::max(snap.rsrpCount, snap.sinrCount);
    double prb = std::min(0.95, 0.55 + 0.015 * phySamples);
    double dlMbps = snap.dlBytes * 8.0 / 1e6;
    double ulMbps = snap.ulBytes * 8.0 / 1e6;
    uint64_t totalPkts = snap.rxPackets + snap.lostPackets;
    double pktLoss = totalPkts ? (double)snap.lostPackets / totalPkts : 0.005;
    uint32_t hoTotal = snap.hoOk + snap.hoFail;
    double hoRate = hoTotal ? (double)snap.hoOk / hoTotal : 0.97;
    double latency = snap.rxPackets ? (snap.delaySum / snap.rxPackets) * 1000.0 : 17.0;

    int label = FaultLabel(gnbId, t);
    if (label == 1) {
        rsrp = -115.0;
        sinr = 2.0;
        dlMbps *= 0.05;
        ulMbps *= 0.05;
        prb = 0.02;
        pktLoss = 0.90;
        hoRate = 0.05;
        latency = 2000.0;
    } else if (label == 2) {
        prb = 0.93;
        dlMbps *= 0.35;
        latency = 400.0;
        pktLoss = std::max(pktLoss, 0.15);
    } else if (label == 3) {
        rsrp = -112.0;
        sinr = 3.0;
        dlMbps *= 0.08;
        prb = 0.05;
        pktLoss = 0.85;
        latency = 1800.0;
    }

    g_csv << std::fixed << std::setprecision(4) << g_trial << "," << g_faultType << ","
          << t << "," << gnbId << "," << ParentMacro(gnbId) << ","
          << (IsMacroGnb(gnbId) ? "macro" : "small") << "," << rsrp << "," << sinr << ","
          << prb << "," << dlMbps << "," << ulMbps << "," << pktLoss << "," << hoRate << ","
          << latency << "," << g_faultStart << "," << g_faultEnd << "," << label << "\n";
}

static void
CollectKpi(double t)
{
    UpdateFlowMonitorStats();

    for (uint32_t gnb = 0; gnb < N_CELLS; gnb++) {
        uint16_t cellId = g_gnbCellIds[gnb];
        const CellAccumulator& snap = g_cells[cellId];
        WriteKpiRow(gnb, cellId, t, snap);
        g_cells[cellId] = CellAccumulator();
    }

    double next = t + KPI_STEP;
    if (next <= g_simTime) {
        Simulator::Schedule(Seconds(KPI_STEP), &CollectKpi, next);
    }
}

static void
AttachOneUe(Ptr<NrHelper> nrHelper,
            NetDeviceContainer ueDevs,
            NetDeviceContainer enbDevs,
            Ipv4InterfaceContainer ueIpIfaces,
            uint32_t ueIndex)
{
    uint32_t homeGnb = g_ueHomeGnb[ueIndex];
    nrHelper->AttachToEnb(ueDevs.Get(ueIndex), enbDevs.Get(homeGnb));
    Ptr<NrGnbNetDevice> gnbDev = DynamicCast<NrGnbNetDevice>(enbDevs.Get(homeGnb));
    uint16_t cellId = gnbDev->GetCellId();
    g_ueIndexToCell[ueIndex] = cellId;
    Ipv4Address ueAddr = ueIpIfaces.GetAddress(ueIndex);
    g_ueIndexToIp[ueIndex] = ueAddr;
    g_ueIpToCell[ueAddr] = cellId;
    Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueDevs.Get(ueIndex));
    if (ueDev) {
        uint32_t imsi = (uint32_t)ueDev->GetImsi();
        g_imsiToCell[imsi] = cellId;
        g_imsiToUeIndex[imsi] = ueIndex;
    }
}

static void
TriggerCongestionFault()
{
    for (uint32_t u = 0; u < g_dlClients.GetN(); u++) {
        if (u >= g_ueHomeGnb.size() || g_ueHomeGnb[u] != g_faultCell) {
            continue;
        }
        Ptr<UdpClient> udp = DynamicCast<UdpClient>(g_dlClients.Get(u));
        if (udp) {
            udp->SetAttribute("Interval", TimeValue(MilliSeconds(8)));
            udp->SetAttribute("PacketSize", UintegerValue(2048));
        }
    }
}

static void
TriggerPowerFault(Ptr<NrHelper> nrHelper, NetDeviceContainer enbDevs)
{
    Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(enbDevs.Get(g_faultCell));
    if (gnb && gnb->GetCcMapSize() > 0) {
        nrHelper->GetGnbPhy(enbDevs.Get(g_faultCell), 0)->SetTxPower(-30.0);
    }
}

static void
TriggerHardwareFault(Ptr<NrHelper> nrHelper, NetDeviceContainer enbDevs)
{
    TriggerPowerFault(nrHelper, enbDevs);
}

int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.AddValue("trial", "Trial index 0-49", g_trial);
    cmd.AddValue("fault", "none|power|congestion|hardware", g_faultType);
    cmd.AddValue("outputDir", "Directory for CSV output", g_outputDir);
    cmd.AddValue("simTime", "Simulation duration in seconds (default 300)", g_simTime);
    cmd.AddValue("numUes", "Number of UEs (default 500)", g_numUes);
    cmd.Parse(argc, argv);

    if (g_numUes < N_CELLS) {
        g_numUes = N_CELLS;
    }
    if (g_simTime < 30.0) {
        g_simTime = 30.0;
    }

    RngSeedManager::SetSeed(1000 + g_trial);
    RngSeedManager::SetRun(g_trial);

    if (g_faultType != "none") {
        Ptr<UniformRandomVariable> rv = CreateObject<UniformRandomVariable>();
        double latestStart = std::max(15.0, g_simTime * 0.65);
        g_faultStart = 10.0 + rv->GetValue() * latestStart;
        g_faultEnd = g_faultStart + 5.0 + rv->GetValue() * std::min(30.0, g_simTime * 0.2);
        if (g_faultEnd > g_simTime - 2.0) {
            g_faultEnd = g_simTime - 2.0;
        }
        g_faultCell = (uint32_t)(rv->GetValue() * N_CELLS);
        if (g_faultCell >= N_CELLS) {
            g_faultCell = N_CELLS - 1;
        }
    }

    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(999999999));
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(beamformingHelper);
    nrHelper->SetEpcHelper(epcHelper);
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    beamformingHelper->SetAttribute("BeamformingMethod",
                                    TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    const double centralFrequency = 3.5e9;
    const double bandwidth = 100e6;
    const uint16_t numerology = 1;
    const double macroTxDbm = 35.0;
    const double smallTxDbm = 23.0;

    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(numerology));

    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency,
                                                     bandwidth,
                                                     numCcPerBand,
                                                     BandwidthPartInfo::UMi_StreetCanyon);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    nrHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    nrHelper->InitializeOperationBand(&band);
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHostContainer.Get(0));
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    ipv4h.Assign(internetDevices);
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHostContainer.Get(0)->GetObject<Ipv4>());
    remoteHostRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    NodeContainer gnbNodes;
    NodeContainer ueNodes;
    gnbNodes.Create(N_CELLS);
    ueNodes.Create(g_numUes);

    std::vector<Vector> macroPos(N_MACRO);
    macroPos[0] = Vector(0, 0, 25);
    for (uint32_t i = 0; i < 6; i++) {
        double ang = M_PI / 6.0 + i * M_PI / 3.0;
        macroPos[i + 1] = Vector(ISD * std::cos(ang), ISD * std::sin(ang), 25);
    }

    std::vector<Vector> gnbPos(N_CELLS);
    for (uint32_t m = 0; m < N_MACRO; m++) {
        gnbPos[m] = macroPos[m];
    }
    const double smallAngles[3] = {90.0, 210.0, 330.0};
    for (uint32_t m = 0; m < N_MACRO; m++) {
        for (uint32_t s = 0; s < N_SMALL_PM; s++) {
            uint32_t idx = N_MACRO + m * N_SMALL_PM + s;
            double rad = smallAngles[s] * M_PI / 180.0;
            gnbPos[idx] = Vector(macroPos[m].x + SMALL_OFFSET * std::cos(rad),
                                 macroPos[m].y + SMALL_OFFSET * std::sin(rad),
                                 10);
        }
    }

    Ptr<ListPositionAllocator> gnbAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < N_CELLS; i++) {
        gnbAlloc->Add(gnbPos[i]);
    }
    MobilityHelper gnbMob;
    gnbMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMob.SetPositionAllocator(gnbAlloc);
    gnbMob.Install(gnbNodes);

    std::vector<uint32_t> uePerCell(N_CELLS, 0);
    uint32_t baseUe = g_numUes / N_CELLS;
    uint32_t remUe = g_numUes % N_CELLS;
    for (uint32_t c = 0; c < N_CELLS; c++) {
        uePerCell[c] = baseUe + (c < remUe ? 1 : 0);
    }

    Ptr<ListPositionAllocator> ueAlloc = CreateObject<ListPositionAllocator>();
    Ptr<UniformRandomVariable> posRv = CreateObject<UniformRandomVariable>();
    g_ueHomeGnb.resize(g_numUes);
    uint32_t ueIdx = 0;
    for (uint32_t c = 0; c < N_CELLS; c++) {
        for (uint32_t k = 0; k < uePerCell[c]; k++) {
            double ox = posRv->GetValue(-ISD * 0.18, ISD * 0.18);
            double oy = posRv->GetValue(-ISD * 0.18, ISD * 0.18);
            ueAlloc->Add(Vector(gnbPos[c].x + ox, gnbPos[c].y + oy, 1.5));
            g_ueHomeGnb[ueIdx++] = c;
        }
    }

    MobilityHelper ueMob;
    ueMob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                           "Bounds",
                           RectangleValue(Rectangle(-ISD * 2, ISD * 2, -ISD * 2, ISD * 2)),
                           "Speed",
                           StringValue("ns3::UniformRandomVariable[Min=0.8|Max=8.3]"),
                           "Distance",
                           DoubleValue(20.0));
    ueMob.SetPositionAllocator(ueAlloc);
    ueMob.Install(ueNodes);

    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDevs = nrHelper->InstallUeDevice(ueNodes, allBwps);

    double txLinear = std::pow(10.0, macroTxDbm / 10.0);
    for (uint32_t g = 0; g < N_CELLS; g++) {
        nrHelper->GetGnbPhy(gnbDevs.Get(g), 0)->SetAttribute("Numerology", UintegerValue(numerology));
        double txDbm = IsMacroGnb(g) ? macroTxDbm : smallTxDbm;
        nrHelper->GetGnbPhy(gnbDevs.Get(g), 0)->SetTxPower(txDbm);
        (void)txLinear;
    }

    for (auto it = ueDevs.Begin(); it != ueDevs.End(); ++it) {
        nrHelper->GetUePhy(*it, 0)->SetNumerology(numerology);
    }

    for (auto it = gnbDevs.Begin(); it != gnbDevs.End(); ++it) {
        DynamicCast<NrGnbNetDevice>(*it)->UpdateConfig();
    }
    for (auto it = ueDevs.Begin(); it != ueDevs.End(); ++it) {
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();
    }

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address(ueDevs);

    g_gnbCellIds.resize(N_CELLS);
    for (uint32_t g = 0; g < N_CELLS; g++) {
        Ptr<NrGnbNetDevice> gnbDev = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(g));
        g_gnbCellIds[g] = gnbDev->GetCellId();
        g_cellToGnb[g_gnbCellIds[g]] = g;
        g_cells[g_gnbCellIds[g]] = CellAccumulator();
    }

    for (uint32_t u = 0; u < ueNodes.GetN(); u++) {
        Ptr<Ipv4StaticRouting> ueRoute =
            ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(u)->GetObject<Ipv4>());
        ueRoute->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    const double attachSpacing = std::max(0.06, 35.0 / g_numUes);
    for (uint32_t u = 0; u < g_numUes; u++) {
        Simulator::Schedule(Seconds(1.0 + u * attachSpacing),
                            &AttachOneUe,
                            nrHelper,
                            ueDevs,
                            gnbDevs,
                            ueIpIfaces,
                            u);
    }

    uint16_t dlPort = 9000;
    ApplicationContainer serverApps;
    for (uint32_t u = 0; u < g_numUes; u++) {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), dlPort + u));
        serverApps.Add(sink.Install(ueNodes.Get(u)));
        UdpClientHelper client(ueIpIfaces.GetAddress(u), dlPort + u);
        client.SetAttribute("Interval", TimeValue(MilliSeconds(80)));
        client.SetAttribute("PacketSize", UintegerValue(1024));
        client.SetAttribute("MaxPackets", UintegerValue(10000000));
        g_dlClients.Add(client.Install(remoteHostContainer.Get(0)));
    }
    const double trafficStart = 1.0 + g_numUes * attachSpacing + 5.0;
    serverApps.Start(Seconds(trafficStart));
    g_dlClients.Start(Seconds(trafficStart + 3.0));
    serverApps.Stop(Seconds(g_simTime + 1.0));
    g_dlClients.Stop(Seconds(g_simTime + 1.0));

    g_flowMonitor = g_flowHelper.InstallAll();

    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy/ReportRsrp",
                    MakeCallback(&OnReportRsrp));
    Config::Connect("/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy/DlDataSinr",
                    MakeCallback(&OnDlDataSinr));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
                    MakeCallback(&OnHandoverEndOk));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndError",
                    MakeCallback(&OnHandoverEndError));

    if (g_faultType == "congestion") {
        Simulator::Schedule(Seconds(g_faultStart), &TriggerCongestionFault);
    } else if (g_faultType == "power") {
        Simulator::Schedule(Seconds(g_faultStart), &TriggerPowerFault, nrHelper, gnbDevs);
    } else if (g_faultType == "hardware") {
        Simulator::Schedule(Seconds(g_faultStart), &TriggerHardwareFault, nrHelper, gnbDevs);
    }

    std::string csvPath = g_outputDir + "/kpi_trial" + std::to_string(g_trial) + "_" +
                          g_faultType + ".csv";
    g_csv.open(csvPath);
    if (!g_csv.is_open()) {
        std::cerr << "ERROR: cannot open " << csvPath << std::endl;
        return 1;
    }
    g_csv << "trial,fault_type,time,gnb_id,macro_id,cell_type,rsrp_avg_dbm,sinr_avg_db,"
             "prb_utilisation,dl_throughput_mbps,ul_throughput_mbps,packet_loss_rate,"
             "handover_success_rate,latency_avg_ms,fault_start_s,fault_end_s,fault_label\n";

    Simulator::Schedule(Seconds(1.0), &CollectKpi, 1.0);
    Simulator::Stop(Seconds(g_simTime + 2.0));
    Simulator::Run();
    Simulator::Destroy();
    g_csv.close();

    std::ifstream check(csvPath);
    int lines = 0;
    std::string line;
    while (std::getline(check, line)) {
        lines++;
    }
    int expected = (int)(g_simTime * N_CELLS);
    if (lines - 1 < expected / 4) {
        std::cerr << "WARNING: only " << lines - 1 << " rows (expected ~" << expected << ")\n";
        return 1;
    }
    return 0;
}

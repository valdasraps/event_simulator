import FWCore.ParameterSet.Config as cms

process = cms.Process("GenDump")

process.source = cms.Source("PoolSource",
    # fileNames = cms.untracked.vstring("file:PPD-Run3Summer23GS-00003.root")
    fileNames = cms.untracked.vstring("/store/mc/Run3Summer23GS/MinBias_TuneCP5_13p6TeV-pythia8/GEN-SIM/130X_mcRun3_2023_realistic_forMB_v1-v3/710000/01db3eaa-abce-47d8-99ad-39d2e4b6e178.root")
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

process.load("FWCore.MessageService.MessageLogger_cfi")

process.genDumper = cms.EDAnalyzer("GenParticleDumper",
    inputTag = cms.InputTag("genParticles", "", "SIM"),
    fileName = cms.string("gen_particles.csv")
)

process.p = cms.Path(process.genDumper)

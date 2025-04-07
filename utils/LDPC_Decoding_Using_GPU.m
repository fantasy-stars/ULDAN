

% Use ldpcQuasiCyclicMatrix to create a parity-check matrix
load("LDPCExamplePrototypeMatrix.mat","P"); % A prototype matrix from the 5G standard
blockSize = 384;
H = ldpcQuasiCyclicMatrix(blockSize, P);
encoderCfg = ldpcEncoderConfig(H);
decoderCfg1 = ldpcDecoderConfig(encoderCfg); % The default algorithm is "bp"
decoderCfg2 = ldpcDecoderConfig(encoderCfg,"norm-min-sum");

M = 4; % Modulation order (QPSK)
snr = [-2 -1.5 -1];
numFramesPerCall = 50;
numCalls = 40;
maxNumIter = 20;
s = rng(1235); % Fix random seed
errRate = zeros(length(snr),2);

for ii = 1:length(snr)
    ttlErr = [0 0];
    noiseVariance = 1/10^(snr(ii)/10);
    for counter = 1:numCalls
        data = logical(randi([0 1],encoderCfg.NumInformationBits,numFramesPerCall));

        % Transmit and receive LDPC coded signal data
        encData = ldpcEncode(data,encoderCfg);
        modSig = pskmod(encData,M,pi/4,'InputType','bit');
        rxSig = awgn(modSig,snr(ii),'measured');
        demodSig = gpuArray(pskdemod(rxSig,M,pi/4,...
            'OutputType','approxllr','NoiseVariance',noiseVariance));

        % Decode and update number of bit errors

        % Using bp
        rxBits1 = ldpcDecode(demodSig,decoderCfg1,maxNumIter);
        numErr1 = biterr(data,rxBits1);

        % Using norm-min-sum
        rxBits2 = ldpcDecode(demodSig,decoderCfg2,maxNumIter);
        numErr2 = biterr(data,rxBits2);

        ttlErr = ttlErr + [numErr1 numErr2];
    end
    ttlBits = numCalls*numel(rxBits1);
    
    errRate(ii,:) = ttlErr/ttlBits;
end

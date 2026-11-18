function milliseconds = detection_time(elapsedSeconds, nSamples)
% Mean time to produce one handover decision, in milliseconds.

milliseconds = 1000 * elapsedSeconds / max(nSamples, 1);
end

function [X, Y] = lineFollower2(R, G, B)
%LINEFOLLOWER2  Accurate angle line-follower using 3-strip centroid tangent.
%
%   Computes the column centroid of the red line in three horizontal strips,
%   then forms a direction vector from the near strip toward the farthest
%   visible strip.  This vector is the tangent of the line — no quantization.
%
%   All operations use fixed-size arrays only (no find/sort/logical indexing
%   of pixel data) — fully compatible with Simulink MATLAB Function blocks.
%
%   Inputs:  R, G, B  -- uint8 arrays, 120x160
%   Outputs: X, Y     -- unit vector  (X: forward/up-in-image,  Y: right)
%
%   Strip layout:
%     Strip A (far):   rows   1-40   centroid row = 20
%     Strip B (mid):   rows  41-80   centroid row = 60
%     Strip C (near):  rows  81-120  centroid row = 100
%   Drone is at image centre: (row=60, col=80)
%
%   Angle accuracy derivation:
%     For a line at angle theta from vertical (theta=0 = straight ahead),
%     a 1-row rise in the image corresponds to a tan(theta)-column shift.
%     The column difference between strip A and strip C centroids over their
%     80-row separation directly encodes tan(theta), so:
%       output angle = atan2(cxFar - cxNear, 80)  [exact, no quantization]

    %% 1. Red-pixel mask — binary, 120x160, fully fixed-size
    r = double(R);  g = double(G);  b = double(B);
    mask = double((r > 180) & (g < 100) & (b < 100));

    if sum(sum(mask)) < 20
        X = 1.0;  Y = 0.0;  return;
    end

    %% 2. Column projections for the three fixed strips
    %   sum(mask(a:b, :), 1) always produces 1x160 — Simulink-safe because
    %   the row bounds are literal constants.
    colIdx = 1:160;                       % 1x160, compile-time constant

    cpFar  = sum(mask(1:40,   :), 1);     % strip A: rows 1-40   (farthest)
    cpMid  = sum(mask(41:80,  :), 1);     % strip B: rows 41-80  (middle)
    cpNear = sum(mask(81:120, :), 1);     % strip C: rows 81-120 (nearest)

    totFar  = sum(cpFar);                 % scalar
    totMid  = sum(cpMid);                 % scalar
    totNear = sum(cpNear);                % scalar

    %% 3. Column centroid of each strip — one scalar per strip
    %   sum(cp .* colIdx) / tot  is fully scalar; colIdx is a fixed 1x160
    %   multiplied element-wise, so the result is always 1x1.
    MIN_PIX = 30.0;

    if totNear >= MIN_PIX
        cxNear = sum(cpNear .* colIdx) / totNear;
    else
        cxNear = 80.0;
    end

    if totMid >= MIN_PIX
        cxMid = sum(cpMid .* colIdx) / totMid;
    else
        cxMid = 80.0;
    end

    if totFar >= MIN_PIX
        cxFar = sum(cpFar .* colIdx) / totFar;
    else
        cxFar = 80.0;
    end

    %% 4. Steering vector: from near strip centroid toward farthest visible strip
    %
    %   steerX = row separation (always positive = always moving forward)
    %   steerY = column shift   (= cxFar - cxNear = lateral steering)
    %
    %   Cases ordered by how much look-ahead is available:
    %
    %   Case 1 — near + far both visible:
    %     steerX = 80  (rows 100 -> 20, 80 rows of look-ahead)
    %     steerY = cxFar - cxNear
    %     Maximum achievable angle: atan2(160, 80) = 63 deg
    %
    %   Case 2 — near + mid only (far off-screen; e.g. very steep turn):
    %     steerX = 40  (rows 100 -> 60, 40 rows of look-ahead)
    %     steerY = cxMid - cxNear
    %
    %   Case 3 — near only (turn is sharper than the camera field of view):
    %     Lateral correction: steer toward where the line IS relative to centre.
    %     steerX = 40 (forward bias ensures the drone keeps moving)
    %     steerY = cxNear - 80
    %
    %   Case 4 / 5 — near missing, only mid or far visible:
    %     Line has moved ahead of drone (e.g. after a gap), steer toward it.

    if totFar >= MIN_PIX && totNear >= MIN_PIX
        steerX = 80.0;
        steerY = cxFar - cxNear;

    elseif totMid >= MIN_PIX && totNear >= MIN_PIX
        steerX = 40.0;
        steerY = cxMid - cxNear;

    elseif totNear >= MIN_PIX
        steerX = 40.0;
        steerY = cxNear - 80.0;

    elseif totMid >= MIN_PIX
        steerX = 40.0;
        steerY = cxMid - 80.0;

    elseif totFar >= MIN_PIX
        steerX = 80.0;
        steerY = cxFar - 80.0;

    else
        X = 1.0;  Y = 0.0;  return;
    end

    %% 5. Normalise to unit vector
    mag = sqrt(steerX^2 + steerY^2);
    if mag < 1e-9
        X = 1.0;  Y = 0.0;
    else
        X = steerX / mag;
        Y = steerY / mag;
    end
end

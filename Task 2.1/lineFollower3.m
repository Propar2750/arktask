function [X, Y] = lineFollower3(R, G, B)
%LINEFOLLOWER3  High-accuracy line follower for any turn angle (incl. 150°+).
%
%   Uses global + perpendicular centroid selection.  At a sharp corner the
%   global centroid of all red pixels is offset toward the new branch;
%   clamping the heading change to ±90° per frame ensures the drone
%   commits to the turn and cannot reverse.
%
%   Fully direction-independent — no axis is assumed as "forward."
%   All arrays are fixed-size.  Simulink MATLAB Function safe.
%
%   Inputs:  R, G, B  -- uint8 arrays, 120x160
%   Outputs: X, Y     -- unit vector  (X: up-in-image,  Y: right-in-image)

    % ---- Hyperparameter ----
    K = 10;   % history buffer length — change this value to tune clamping
    % ------------------------

    % Persistent heading — remembers which way the drone is going
    persistent dirX dirY;
    % History buffer: last K output directions for clamp reference
    persistent histX histY histN;
    if isempty(dirX)
        dirX = 1.0;   dirY = 0.0;   % will adapt in the first few frames
        histX = zeros(1, K);
        histY = zeros(1, K);
        histN = 0;                   % how many entries filled so far
    end

    %% 1. Red mask
    r = double(R);  g = double(G);  b = double(B);
    mask = double((r > 180) & (g < 100) & (b < 100));

    totalRed = sum(sum(mask));
    if totalRed < 15
        X = dirX;  Y = dirY;  return;     % keep going in last known direction
    end

    %% 2. Global centroid of ALL red pixels (fully fixed-size)
    colIdx = 1:160;              % 1x160
    rowIdx = (1:120)';           % 120x1

    colProj = sum(mask, 1);      % 1x160 — total red per column
    rowProj = sum(mask, 2);      % 120x1 — total red per row

    gcCol = sum(colProj .* colIdx) / totalRed;    % scalar
    gcRow = sum(rowProj .* rowIdx) / totalRed;    % scalar

    imgCR = 60.0;
    imgCC = 80.0;

    %% 3. Compute the "ahead centroid" — centroid of only the red pixels
    %   in the forward half of the image relative to the heading.
    %
    %   For each pixel (row, col), its signed distance along the heading is:
    %     proj = (imgCR - row)*dirX + (col - imgCC)*dirY
    %   We want pixels with proj < 0 (they are behind) to be excluded,
    %   and keep only proj >= 0 (ahead or beside the drone).
    %
    %   To do this with fixed-size ops, compute a weight mask:
    %     wt(row, col) = max(0, proj(row, col))
    %   Then the weighted centroid of mask.*wt gives us the ahead centroid.
    %   The weights also favour pixels farther ahead — better look-ahead.

    % Build proj for every pixel using outer-product broadcasting
    % projRow(i) = (imgCR - i)*dirX,  projCol(j) = (j - imgCC)*dirY
    % proj(i,j) = projRow(i) + projCol(j)
    projRow = (imgCR - rowIdx) * dirX;     % 120x1
    projCol = (colIdx - imgCC) * dirY;     % 1x160

    % Weighted mask: only pixels ahead of the drone, weighted by distance
    % ahead.  We add 1.0 to make side-pixels (proj≈0) contribute slightly.
    aheadWt = mask .* max(0, projRow + projCol + 1.0);    % 120x160

    aheadTot = sum(sum(aheadWt));

    if aheadTot > 1.0
        aheadCol = sum(sum(aheadWt, 1) .* colIdx) / aheadTot;
        aheadRow = sum(sum(aheadWt, 2) .* rowIdx) / aheadTot;
    else
        % Nothing ahead — use global centroid
        aheadCol = gcCol;
        aheadRow = gcRow;
    end

    %% 4. Compute the "perp centroid" — centroid of red pixels with the
    %   largest perpendicular offset from the heading line.
    %   At a corner, the new branch has large perp offset → pulls the
    %   centroid to the correct side.
    %
    %   perpDist(i,j) = (imgCR - i)*(-dirY) + (j - imgCC)*(dirX)
    %   We use |perpDist| as weight to emphasize pixels far from the line.

    perpRow = (imgCR - rowIdx) * (-dirY);  % 120x1
    perpCol = (colIdx - imgCC) * dirX;     % 1x160

    perpWt = mask .* abs(perpRow + perpCol);    % 120x160
    perpTot = sum(sum(perpWt));

    if perpTot > 1.0
        perpCol_c = sum(sum(perpWt, 1) .* colIdx) / perpTot;
        perpRow_c = sum(sum(perpWt, 2) .* rowIdx) / perpTot;
    else
        perpCol_c = gcCol;
        perpRow_c = gcRow;
    end

    %% 5. Blend ahead centroid and perp centroid
    %   Normally the ahead centroid dominates (straight line / gentle curve).
    %   At a sharp corner, most pixels are behind → aheadTot is small
    %   relative to totalRed.  Use this ratio to blend in the perp centroid
    %   which correctly identifies the new branch side.
    behindRatio = 1.0 - min(aheadTot / max(totalRed, 1.0), 1.0);
    % behindRatio ≈ 0 for straight lines, ≈ 0.5-0.8 for sharp corners

    blendCol = (1.0 - behindRatio) * aheadCol + behindRatio * perpCol_c;
    blendRow = (1.0 - behindRatio) * aheadRow + behindRatio * perpRow_c;

    %% 6. Steering vector from image centre toward blended target
    steerX = imgCR - blendRow;
    steerY = blendCol - imgCC;

    %% 7. Clamp: output must be within ±90° of every one of the last K
    %   directions.  Find the historical direction with the largest angular
    %   difference from the raw steer angle; if that exceeds 90° clamp it.
    rawAng  = atan2(steerY, steerX);
    outAng  = rawAng;           % default: no clamping

    MAX_TURN = pi / 2;          % ±90° → 180° cone

    if histN > 0
        nUse = min(histN, K);
        worstDiff = 0.0;
        worstRef  = rawAng;
        worstSign = 1.0;
        for k = 1:nUse
            hAng = atan2(histY(k), histX(k));
            d = rawAng - hAng;
            if d > pi
                d = d - 2*pi;
            elseif d < -pi
                d = d + 2*pi;
            end
            if abs(d) > abs(worstDiff)
                worstDiff = d;
                worstRef  = hAng;
            end
        end
        % Clamp if the largest difference exceeds the limit
        if worstDiff > MAX_TURN
            outAng = worstRef + MAX_TURN;
        elseif worstDiff < -MAX_TURN
            outAng = worstRef - MAX_TURN;
        end
    end
    X = cos(outAng);
    Y = sin(outAng);

    %% 8. Update history buffer (circular, fixed-size K)
    slot = mod(histN, K) + 1;    % 1-based slot index, wraps at K
    histX(slot) = X;
    histY(slot) = Y;
    histN = histN + 1;

    %% 9. Smooth-update persistent heading
    alpha = 0.7;
    newDirX = alpha * X + (1 - alpha) * dirX;
    newDirY = alpha * Y + (1 - alpha) * dirY;
    magD = sqrt(newDirX^2 + newDirY^2);
    if magD > 1e-9
        dirX = newDirX / magD;
        dirY = newDirY / magD;
    end
end

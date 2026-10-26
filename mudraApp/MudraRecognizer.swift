import AVFoundation
import CoreML
import SwiftUI
import UIKit
import Vision
import Combine

// Add MudraClassifier.mlpackage to your Xcode target (Xcode generates `MudraClassifier`)
// and add NSCameraUsageDescription to Info.plist.

struct MudraPrediction: Equatable {
    let name: String
    let confidence: Float
    let handCount: Int
}

final class MudraRecognizer: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    @Published var prediction: MudraPrediction?   // nil = no hand / not confident
    let session = AVCaptureSession()

    // Same list as DOUBLE_HAND in train_mudra.py.
    private let doubleHand: Set<String> = [
        "Anjali", "Kapotham", "Karkatta", "Swastikam", "Pushpaputa", "Shivalinga",
        "Katakavardhana", "Kartariswastika", "Sakata", "Shanka", "Chakra", "Samputa",
        "Pasha", "Kilaka", "Matsya", "Kurma", "Varaha", "Garuda", "Nagabandha",
        "Khatva", "Berunda",
    ]

    private let queue = DispatchQueue(label: "mudra.video")
    private let handRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 2
        return r
    }()
    private let classifyRequest: VNCoreMLRequest
    private var history: [[String: Float]] = []
    private let historyLength = 8        // frames averaged to stop flicker
    private let minConfidence: Float = 0.6
    private var frameCounter = 0
    private var usingFrontCamera = true

    override init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let core = try! MudraClassifier(configuration: config).model
        let vnModel = try! VNCoreMLModel(for: core)
        classifyRequest = VNCoreMLRequest(model: vnModel)
        classifyRequest.imageCropAndScaleOption = .scaleFill
        super.init()
    }

    // MARK: Live camera

    func startCamera(front: Bool = true) {
        usingFrontCamera = front
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            self.queue.async { self.configureSession(front: front) }
        }
    }

    func stopCamera() { queue.async { self.session.stopRunning() } }

    private func configureSession(front: Bool) {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video,
                                                 position: front ? .front : .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else { session.commitConfiguration(); return }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
        session.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        frameCounter += 1
        guard frameCounter % 2 == 0,                       // ~15 fps is plenty
              let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Portrait app: back camera buffers are rotated 90°, front camera is also mirrored.
        let orientation: CGImagePropertyOrientation = usingFrontCamera ? .leftMirrored : .right
        let image = CIImage(cvPixelBuffer: buffer).oriented(orientation)

        guard let result = recognize(image) else {
            history.removeAll()
            DispatchQueue.main.async { self.prediction = nil }
            return
        }
        history.append(result.scores)
        if history.count > historyLength { history.removeFirst() }

        // Average the recent frames.
        var avg: [String: Float] = [:]
        for h in history { for (k, v) in h { avg[k, default: 0] += v / Float(history.count) } }
        let best = avg.max { $0.value < $1.value }
        let output: MudraPrediction? = best.flatMap {
            $0.value >= minConfidence ? MudraPrediction(name: $0.key, confidence: $0.value,
                                                         handCount: result.handCount) : nil
        }
        DispatchQueue.main.async { self.prediction = output }
    }

    // MARK: Single photo

    func classify(photo: UIImage) -> MudraPrediction? {
        guard let ci = CIImage(image: photo) else { return nil }
        let image = ci.oriented(CGImagePropertyOrientation(photo.imageOrientation))
        guard let r = recognize(image),
              let best = r.scores.max(by: { $0.value < $1.value }) else { return nil }
        return MudraPrediction(name: best.key, confidence: best.value, handCount: r.handCount)
    }

    // MARK: Core pipeline: find hands -> square crop -> classify -> re-weight by hand count

    private func recognize(_ image: CIImage) -> (scores: [String: Float], handCount: Int)? {
        let extent = image.extent
        let handler = VNImageRequestHandler(ciImage: image)
        guard (try? handler.perform([handRequest])) != nil,
              let hands = handRequest.results, !hands.isEmpty else { return nil }

        // Bounding box of all confident joints from every detected hand (normalised).
        var minX = CGFloat.greatestFiniteMagnitude, minY = minX
        var maxX = -CGFloat.greatestFiniteMagnitude, maxY = maxX
        for hand in hands {
            guard let pts = try? hand.recognizedPoints(.all) else { continue }
            for (_, p) in pts where p.confidence > 0.3 {
                minX = min(minX, p.location.x); maxX = max(maxX, p.location.x)
                minY = min(minY, p.location.y); maxY = max(maxY, p.location.y)
            }
        }
        guard minX < maxX, minY < maxY else { return nil }

        let w = (maxX - minX) * extent.width, h = (maxY - minY) * extent.height
        let cx = extent.minX + (minX + maxX) / 2 * extent.width
        let cy = extent.minY + (minY + maxY) / 2 * extent.height
        let side = min(max(w, h) * 1.35, min(extent.width, extent.height))   // 35% padding
        let x = max(extent.minX, min(cx - side / 2, extent.maxX - side))
        let y = max(extent.minY, min(cy - side / 2, extent.maxY - side))
        let crop = image
            .cropped(to: CGRect(x: x, y: y, width: side, height: side))
            .transformed(by: CGAffineTransform(translationX: -x, y: -y))

        let classifyHandler = VNImageRequestHandler(ciImage: crop)
        guard (try? classifyHandler.perform([classifyRequest])) != nil,
              let obs = classifyRequest.results as? [VNClassificationObservation] else { return nil }

        // Soft prior from the hand count. Not a hard mask: when palms touch (Anjali, Samputa...)
        // Vision sometimes finds only one hand.
        let count = hands.count
        var scores: [String: Float] = [:]
        var total: Float = 0
        for o in obs {
            let isDouble = doubleHand.contains(o.identifier)
            let weight: Float = count >= 2 ? (isDouble ? 1 : 0.05) : (isDouble ? 0.3 : 1)
            scores[o.identifier] = o.confidence * weight
            total += o.confidence * weight
        }
        guard total > 0 else { return nil }
        for k in scores.keys { scores[k]! /= total }
        return (scores, count)
    }
}

extension CGImagePropertyOrientation {
    init(_ o: UIImage.Orientation) {
        switch o {
        case .up: self = .up
        case .down: self = .down
        case .left: self = .left
        case .right: self = .right
        case .upMirrored: self = .upMirrored
        case .downMirrored: self = .downMirrored
        case .leftMirrored: self = .leftMirrored
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}

// MARK: - Minimal SwiftUI screen

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

struct MudraCameraView: View {
    @StateObject private var recognizer = MudraRecognizer()

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraPreview(session: recognizer.session).ignoresSafeArea()
            VStack(spacing: 4) {
                Text(recognizer.prediction?.name ?? "Show a mudra")
                    .font(.title.bold())
                if let p = recognizer.prediction {
                    Text("\(Int(p.confidence * 100))% · \(p.handCount) hand\(p.handCount == 1 ? "" : "s")")
                        .font(.subheadline)
                }
            }
            .padding(.horizontal, 24).padding(.vertical, 14)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            .padding(.bottom, 40)
        }
        .onAppear { recognizer.startCamera(front: true) }
        .onDisappear { recognizer.stopCamera() }
    }
}

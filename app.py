from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import os
import warnings
import tempfile

# 경고 억제
warnings.filterwarnings('ignore', message='Trying to estimate tuning from empty frequency set')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

app = Flask(__name__)
CORS(app)

class ScreamAnalyzer:
    def __init__(self, model_path='scream_detector.pkl'):
        """학습된 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 없음: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"✓ 모델 로드 완료: {model_path}")
    
    def extract_features(self, audio_segment, sr):
        """오디오 세그먼트에서 특징 추출"""
        try:
            # 무음 구간 체크
            if np.max(np.abs(audio_segment)) < 0.01:
                return None
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 추가 특징
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=sr))
            rms = np.mean(librosa.feature.rms(y=audio_segment))
            
            # 피치 관련
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
            
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [zcr, spectral_centroid, spectral_rolloff, rms],
                chroma_mean
            ])
            
            return features
        
        except Exception as e:
            print(f"특징 추출 실패: {e}")
            return None
    
    def analyze_audio(self, audio_path, window_size=1.5, hop_size=1.0, threshold=0.6):
        """오디오 파일에서 비명 감지"""
        print(f"\n분석 중: {audio_path}")
        
        # 오디오 로드
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr
            print(f"오디오 길이: {duration:.1f}초")
        except Exception as e:
            print(f"오디오 로드 실패: {e}")
            return []
        
        # 슬라이딩 윈도우 분석
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        scream_detections = []
        all_predictions = []  # 디버깅용: 모든 예측 저장
        total_windows = (len(y) - window_samples) // hop_samples
        
        print(f"분석 시작 (총 {total_windows}개 구간)")
        
        for idx, i in enumerate(range(0, len(y) - window_samples, hop_samples)):
            segment = y[i:i+window_samples]
            
            # 특징 추출
            features = self.extract_features(segment, sr)
            if features is None:
                continue
            
            # 예측
            features_2d = features.reshape(1, -1)
            prediction = self.model.predict(features_2d)[0]
            probability = self.model.predict_proba(features_2d)[0]
            
            # 비명 확률
            scream_prob = probability[1]
            
            # 디버깅: 모든 예측 기록
            all_predictions.append({
                'time': i / sr,
                'prediction': int(prediction),
                'scream_prob': float(scream_prob)
            })
            
            if prediction == 1 and scream_prob >= threshold:
                start_time = i / sr
                end_time = (i + window_samples) / sr
                scream_detections.append([start_time, end_time, float(scream_prob)])
        
        # 디버깅 정보 출력
        print(f"✓ 분석 완료")
        print(f"총 분석 구간: {len(all_predictions)}개")
        print(f"비명 예측(prediction=1): {sum(1 for p in all_predictions if p['prediction'] == 1)}개")
        print(f"임계값 이상 구간: {len(scream_detections)}개")
        
        # 상위 5개 확률 출력
        top_predictions = sorted(all_predictions, key=lambda x: x['scream_prob'], reverse=True)[:5]
        print("상위 5개 비명 확률:")
        for i, pred in enumerate(top_predictions, 1):
            print(f"  {i}. 시간: {pred['time']:.1f}초, 확률: {pred['scream_prob']:.2%}, 예측: {pred['prediction']}")
        
        # 인접한 구간 병합
        merged = self.merge_detections(scream_detections, merge_gap=1.5)
        
        return merged
    
    def merge_detections(self, detections, merge_gap=1.5):
        """인접한 비명 구간 병합"""
        if not detections:
            return []
        
        # 시작 시간으로 정렬
        detections = sorted(detections, key=lambda x: x[0])
        
        merged = []
        current_start, current_end, current_prob = detections[0]
        
        for start, end, prob in detections[1:]:
            if start - current_end <= merge_gap:
                current_end = end
                current_prob = max(current_prob, prob)
            else:
                merged.append([current_start, current_end, current_prob])
                current_start, current_end, current_prob = start, end, prob
        
        merged.append([current_start, current_end, current_prob])
        
        return merged


# 전역 analyzer 인스턴스
analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = ScreamAnalyzer('scream_detector.pkl')
    return analyzer


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'message': '비명 감지 API 서버가 실행 중입니다',
        'endpoints': {
            '/analyze': 'POST - 오디오 파일 분석'
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 파일 확인
        if 'audio' not in request.files:
            return jsonify({'error': '오디오 파일이 없습니다'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
        
        # 파라미터 가져오기
        window_size = float(request.form.get('window_size', 1.5))
        hop_size = float(request.form.get('hop_size', 1.0))
        threshold = float(request.form.get('threshold', 0.6))
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # 분석 실행
            analyzer = get_analyzer()
            detections = analyzer.analyze_audio(
                temp_path,
                window_size=window_size,
                hop_size=hop_size,
                threshold=threshold
            )
            
            return jsonify({
                'success': True,
                'detections': detections,
                'count': len(detections)
            })
        
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"에러 발생: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

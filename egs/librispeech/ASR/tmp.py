import gzip
import json
import os

# --- 설정 ---
field = "_eval_clean"
input_file = f"data/ko/manifests/ksponspeech_recordings{field}.jsonl.gz"
output_file = f"data/ko/manifests/ksponspeech_noisy_recordings{field}.jsonl.gz"
key_to_find = "storage_path"
new_path_value = f"data/en/fbank/musan_feats{field}/"
# ------------

def update_nested_key(obj, key, new_value):
    """
    중첩된 딕셔너리(dict) 또는 리스트(list)를 재귀적으로 탐색하며
    모든 'key'의 값을 'new_value'로 변경합니다.
    """
    if isinstance(obj, dict):
        # 1. 객체가 딕셔너리인 경우
        for k, v in obj.items():
            if k == key:
                # ⭐️ 키가 일치하면 값을 바로 변경
                filename = os.path.basename(v)
                obj[k] = os.path.join(new_value, filename)
            elif isinstance(v, (dict, list)):
                # 값이 딕셔너리나 리스트이면 재귀 호출
                update_nested_key(v, key, new_value)

    elif isinstance(obj, list):
        # 2. 객체가 리스트인 경우
        for item in obj:
            if isinstance(item, (dict, list)):
                # 리스트의 항목이 딕셔너리나 리스트이면 재귀 호출
                update_nested_key(item, key, new_value)

# --- 메인 실행 로직 ---
temp_output_file = output_file + ".tmp"

try:
    with gzip.open(temp_output_file, 'wt', encoding='utf-8') as f_out:
        with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
            
            line_count = 0
            for line in f_in:
                if not line.strip():
                    continue
                    
                data = json.loads(line.strip())
                
                # ⭐️ 단순 대입 대신 재귀 함수 호출
                update_nested_key(data, key_to_find, new_path_value)
                
                # 수정된 데이터를 JSON 문자열로 변환하여 저장
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                line_count += 1

    # 성공 시 임시 파일 이름을 최종 파일명으로 변경
    os.rename(temp_output_file, output_file)
    print(f"성공: 총 {line_count}줄을 처리하여 '{output_file}' 파일에 저장했습니다. (Nested 구조 처리 완료)")

except Exception as e:
    print(f"오류 발생: {e}")
    # 실패 시 임시 파일 삭제
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
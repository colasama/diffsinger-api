"""Stable failure categories for clients; the original error remains untouched."""
import ast
import re
import subprocess


class TaskError(ValueError):
    def __init__(self, message: str, code: str, **context):
        super().__init__(message)
        self.code = code
        self.context = context


def error_info(error: Exception | str, stage: str) -> dict:
    message = str(error)
    context = {}
    code = 'UNKNOWN'
    if isinstance(error, TaskError):
        code, context = error.code, error.context
    elif 'Phonemes missing from HubertFA ' in message:
        code = 'PHONEME_UNSUPPORTED'
        match = re.search(r'vocabulary: (\[[^\n]*\])', message)
        if match:
            try:
                phones = ast.literal_eval(match[1])
                if isinstance(phones, list) and all(isinstance(p, str) for p in phones):
                    context['phonemes'] = phones
            except (ValueError, SyntaxError):
                pass
    elif 'word missing from voicebank dictionary:' in message:
        code = 'LYRIC_WORD_NOT_FOUND'
        context['word'] = message.split('word missing from voicebank dictionary:', 1)[1].split('\n', 1)[0].strip()
    elif 'no singable lyric lines parsed from LRC' in message or 'song has no LRC lyrics' in message:
        code = 'LYRICS_MISSING'
    elif stage == 'fetch' and ('ffprobe' in message or isinstance(error, subprocess.CalledProcessError)
                               and error.cmd and error.cmd[0] == 'ffprobe'):
        code = 'AUDIO_READ_FAILED'
    elif stage == 'fetch' and any(s in message for s in (
        'ConnectError:', 'ConnectTimeout:', 'ReadTimeout:', 'ReadError:', 'RemoteProtocolError:',
        'HTTPStatusError:', 'UNEXPECTED_EOF_WHILE_READING',
    )):
        code = 'SOURCE_UNAVAILABLE'
    # Exception str() omits its type, while legacy saved errors include it.
    elif stage == 'fetch' and type(error).__name__ in {
        'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'ReadError', 'RemoteProtocolError', 'HTTPStatusError',
    }:
        code = 'SOURCE_UNAVAILABLE'
    result = {'code': code, 'stage': stage, 'context': context,
              'retryable': code == 'SOURCE_UNAVAILABLE'}
    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        result['diagnostic'] = error.stderr.decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr
    return result

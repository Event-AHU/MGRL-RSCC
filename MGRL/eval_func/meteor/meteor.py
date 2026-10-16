#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import shutil
import subprocess
import threading

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

class Meteor:

    def __init__(self):
        self.lock = threading.Lock()
        self.meteor_p = None
        java_bin = os.environ.get('JAVA_BIN', 'java')
        meteor_jar = os.path.abspath(os.path.expanduser(os.environ.get(
            'METEOR_JAR', os.path.join(RESOURCE_DIR, 'meteor-1.5.jar'))))
        paraphrase_path = os.path.abspath(os.path.expanduser(os.environ.get(
            'METEOR_PARAPHRASE', os.path.join(RESOURCE_DIR, 'data', 'paraphrase-en.gz'))))
        if shutil.which(java_bin) is None:
            raise FileNotFoundError('Java is required for METEOR; install Java or set JAVA_BIN.')
        for resource_path in (meteor_jar, paraphrase_path):
            if not os.path.isfile(resource_path):
                raise FileNotFoundError(
                    f'Missing METEOR resource: {resource_path}. '
                    'Place the resources in eval_func/meteor or set METEOR_JAR and METEOR_PARAPHRASE.'
                )
        self.env = os.environ.copy()
        self.meteor_cmd = [java_bin, '-Xmx2G', '-jar', meteor_jar, '-', '-', '-stdio', '-l', 'en', '-norm', '-a', paraphrase_path]

        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)

    def compute_score(self, gts, res):

        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in range(len(res)):
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')

        # Collect segment scores
        for i in range(len(res)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line + '\n')
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        process = getattr(self, 'meteor_p', None)
        if process is None:
            return
        with self.lock:
            if process.stdin is not None:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

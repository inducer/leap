set -x
set -e
sed -i s/Instruction/Statement/g $(git ls-files | grep -v insn-to-statement.sh | grep -v .txt )
sed -i s/instruction/statement/g $(git ls-files | grep -v insn-to-statement.sh | grep -v .txt )
sed -i s/INSTRUCTION/STATEMENT/g $(git ls-files | grep -v insn-to-statement.sh | grep -v .txt )
sed -i s/insn/stmt/g $(git ls-files | grep -v insn-to-statement.sh | grep -v .txt )
#patch -p1 < ./stmt-compat-fixes.patch
